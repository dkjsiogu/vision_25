#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth},
  outpost_target_(config_path),
  is_tracking_outpost_(false),
  config_path_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
    is_tracking_outpost_ = false;
    outpost_target_.reset();
  }

  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 检查是否有前哨站装甲板
  bool has_outpost = std::any_of(
    armors.begin(), armors.end(), [](const Armor & a) { return a.name == ArmorName::outpost; });

  // 如果正在追踪前哨站，或者检测到前哨站装甲板，使用前哨站专用追踪
  if (is_tracking_outpost_ || has_outpost) {
    return handle_outpost(armors, t) ? std::list<Target>{outpost_to_target(t)} : std::list<Target>{};
  }

  // 非前哨站目标的正常追踪逻辑

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  // 前哨站：与 sp_vision_25 保持一致
  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  int found_count = 0;
  double min_x = 1e10;  // 画面最左侧
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
    min_x = armor.center.x < min_x ? armor.center.x : min_x;
  }

  if (found_count == 0) return false;

  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x
    )
      continue;

    solver_.solve(armor);

    target_.update(armor);
  }

  return true;
}

bool Tracker::handle_outpost(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  // 提取前哨站装甲板
  std::list<Armor> outpost_armors;
  for (auto & armor : armors) {
    if (armor.name == ArmorName::outpost) {
      solver_.solve(armor);
      outpost_armors.push_back(armor);
    }
  }

  // 如果没有前哨站装甲板
  if (outpost_armors.empty()) {
    // 如果正在追踪前哨站，进行预测
    if (is_tracking_outpost_ && outpost_target_.state() == OutpostState::TRACKING) {
      outpost_target_.predict(t);
      // 检查丢失计数 (这里简化处理，依赖OutpostTarget内部状态)
      // 如果OutpostTarget状态变为非TRACKING，则重置
      if (outpost_target_.state() != OutpostState::TRACKING) {
        tools::logger()->info("[Tracker] Outpost target lost");
        is_tracking_outpost_ = false;
        state_ = "lost";
        return false;
      }
      state_ = "temp_lost";
      return true;  // 仍然返回target用于预测瞄准
    }

    // 否则退出前哨站追踪模式
    if (is_tracking_outpost_) {
      tools::logger()->info("[Tracker] Exit outpost tracking mode");
      is_tracking_outpost_ = false;
      outpost_target_.reset();
    }
    state_ = "lost";
    return false;
  }

  // 有前哨站装甲板，更新追踪
  is_tracking_outpost_ = true;

  // 选择最靠近图像中心的装甲板
  outpost_armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    return cv::norm(a.center - img_center) < cv::norm(b.center - img_center);
  });

  auto & best_armor = outpost_armors.front();

  // 更新前哨站追踪器
  bool tracking = outpost_target_.update(best_armor, t);

  // 更新Tracker状态以匹配OutpostTarget状态
  switch (outpost_target_.state()) {
    case OutpostState::LOST:
      state_ = "lost";
      break;
    case OutpostState::TRACKING:
      state_ = "tracking";
      break;
  }

  return tracking;
}

Target Tracker::outpost_to_target(std::chrono::steady_clock::time_point t) const
{
  // 将OutpostTarget转换为Target以兼容现有Aimer/Planner接口
  auto ekf_x = outpost_target_.ekf_x();
  auto armor_list = outpost_target_.armor_xyza_list();
  auto height_offsets = outpost_target_.height_offsets();  // 获取高度偏移！
  auto initialized_ids = outpost_target_.initialized_ids();  // 获取已初始化装甲板ID！

  // 调试：输出完整的状态信息
  tools::logger()->debug(
    "[Tracker] outpost_to_target: ekf_x=[cx={:.3f}, vx={:.3f}, cy={:.3f}, vy={:.3f}, z={:.3f}, vz={:.3f}, angle={:.3f}, omega={:.3f}, r={:.3f}]",
    ekf_x[0], ekf_x[1], ekf_x[2], ekf_x[3], ekf_x[4], ekf_x[5], ekf_x[6], ekf_x[7], ekf_x[8]);
  tools::logger()->debug(
    "[Tracker] outpost_to_target: armor_list_size={}, initialized_ids_size={}, height_offsets=[{:.3f}, {:.3f}, {:.3f}]",
    armor_list.size(), initialized_ids.size(),
    height_offsets.size() > 0 ? height_offsets[0] : 0.0,
    height_offsets.size() > 1 ? height_offsets[1] : 0.0,
    height_offsets.size() > 2 ? height_offsets[2] : 0.0);
  for (size_t i = 0; i < armor_list.size(); i++) {
    tools::logger()->debug(
      "[Tracker] outpost_to_target: armor[{}]=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
      i, armor_list[i][0], armor_list[i][1], armor_list[i][2], armor_list[i][3]);
  }

  return Target(
    ArmorName::outpost,
    ArmorType::small,
    outpost_target_.priority,
    outpost_target_.jumped,
    outpost_target_.last_id,
    ekf_x,
    armor_list,
    3,  // 前哨站3个装甲板
    t,  // 传递帧时间戳！
    height_offsets,  // 传递高度偏移！
    initialized_ids  // 传递已初始化装甲板ID！
  );
}

}  // namespace auto_aim