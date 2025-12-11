#include "outpost_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

OutpostTarget::OutpostTarget(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  // 读取前哨站扫描配置
  if (yaml["outpost_scan_timeout"]) {
    scan_timeout_ = yaml["outpost_scan_timeout"].as<double>();
  }
  if (yaml["outpost_layer_gap_min"]) {
    layer_gap_min_ = yaml["outpost_layer_gap_min"].as<double>();
  }
  if (yaml["outpost_layer_gap_max"]) {
    layer_gap_max_ = yaml["outpost_layer_gap_max"].as<double>();
  }
  if (yaml["outpost_cluster_threshold"]) {
    cluster_threshold_ = yaml["outpost_cluster_threshold"].as<double>();
  }
  if (yaml["outpost_validation_threshold"]) {
    validation_error_threshold_ = yaml["outpost_validation_threshold"].as<double>();
  }
  if (yaml["outpost_min_observations"]) {
    min_observations_ = yaml["outpost_min_observations"].as<int>();
  }
  if (yaml["outpost_max_temp_lost_count"]) {
    max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  }

  reset();
}

std::string OutpostTarget::state_string() const
{
  switch (state_) {
    case OutpostState::LOST:
      return "lost";
    case OutpostState::SCANNING:
      return "scanning";
    case OutpostState::VALIDATING:
      return "validating";
    case OutpostState::TRACKING:
      return "tracking";
    default:
      return "unknown";
  }
}

void OutpostTarget::reset()
{
  state_ = OutpostState::LOST;
  model_ = OutpostModel{};
  observations_.clear();
  ekf_initialized_ = false;
  consecutive_validation_failures_ = 0;
  temp_lost_count_ = 0;
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  // 检查是否是前哨站装甲板
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  last_update_time_ = t;
  temp_lost_count_ = 0;

  switch (state_) {
    case OutpostState::LOST: {
      // 开始扫描
      state_ = OutpostState::SCANNING;
      scan_start_time_ = t;
      observations_.clear();
      model_ = OutpostModel{};
      model_.z_ref = armor.xyz_in_world[2];  // 设置z参考点
      add_observation(armor, t);
      tools::logger()->info("[OutpostTarget] Start scanning, z_ref={:.3f}", model_.z_ref);
      return false;
    }

    case OutpostState::SCANNING: {
      // 检查超时
      double elapsed = tools::delta_time(t, scan_start_time_);
      if (elapsed > scan_timeout_) {
        tools::logger()->warn("[OutpostTarget] Scan timeout ({:.1f}s), reset", elapsed);
        reset();
        return false;
      }

      add_observation(armor, t);

      // 尝试建模
      if (try_build_model()) {
        state_ = OutpostState::VALIDATING;
        model_.validation_count = 0;
        consecutive_validation_failures_ = 0;
        tools::logger()->info(
          "[OutpostTarget] Model built! layers=[{:.3f}, {:.3f}, {:.3f}], omega={:.2f}",
          model_.layer_z[0], model_.layer_z[1], model_.layer_z[2], model_.omega);
      }
      return false;
    }

    case OutpostState::VALIDATING: {
      if (validate_observation(armor, t)) {
        model_.validation_count++;
        consecutive_validation_failures_ = 0;

        if (model_.validation_count >= validation_required_) {
          state_ = OutpostState::TRACKING;
          init_ekf(armor, t);
          tools::logger()->info("[OutpostTarget] Validation passed, start tracking!");
          return true;
        }
      } else {
        consecutive_validation_failures_++;
        if (consecutive_validation_failures_ > 5) {
          tools::logger()->warn("[OutpostTarget] Validation failed, re-scanning");
          state_ = OutpostState::SCANNING;
          scan_start_time_ = t;
          observations_.clear();
          model_.is_ready = false;
        }
      }
      return false;
    }

    case OutpostState::TRACKING: {
      update_tracking(armor, t);
      return true;
    }

    default:
      return false;
  }
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  if (state_ != OutpostState::TRACKING || !ekf_initialized_) {
    return;
  }

  double dt = tools::delta_time(t, last_update_time_);
  predict(dt);
  last_update_time_ = t;
}

void OutpostTarget::predict(double dt)
{
  if (!ekf_initialized_) return;

  // 状态转移矩阵 (简化版：只预测相位)
  // 状态: [center_x, center_y, z_ref, phase, omega]
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(5, 5);
  F(3, 4) = dt;  // phase += omega * dt

  // 过程噪声
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(5, 5);
  Q(0, 0) = 0.001;  // center_x 方差
  Q(1, 1) = 0.001;  // center_y 方差
  Q(2, 2) = 0.01;   // z_ref 方差 (允许较大漂移)
  Q(3, 3) = 0.01;   // phase 方差
  Q(4, 4) = 0.001;  // omega 方差 (前哨站转速稳定)

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[3] = tools::limit_rad(x_prior[3]);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
}

void OutpostTarget::add_observation(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  OutpostObservation obs;

  // 从装甲板位置推算旋转中心
  double armor_yaw = armor.ypr_in_world[0];
  obs.x = armor.xyz_in_world[0] + model_.radius * std::cos(armor_yaw);
  obs.y = armor.xyz_in_world[1] + model_.radius * std::sin(armor_yaw);
  obs.z = armor.xyz_in_world[2];
  obs.z_relative = obs.z - model_.z_ref;
  obs.armor_yaw = armor_yaw;
  obs.t = t;

  observations_.push_back(obs);

  // 限制观测队列长度
  while (observations_.size() > 30) {
    observations_.pop_front();
  }

  // 更新旋转中心估计 (滑动平均)
  double sum_x = 0, sum_y = 0;
  for (const auto & o : observations_) {
    sum_x += o.x;
    sum_y += o.y;
  }
  model_.center_x = sum_x / observations_.size();
  model_.center_y = sum_y / observations_.size();
}

bool OutpostTarget::try_build_model()
{
  if (observations_.size() < static_cast<size_t>(min_observations_)) {
    return false;
  }

  // 1. 检查角度覆盖
  double coverage = calculate_angle_coverage();
  if (coverage < angle_coverage_required_) {
    return false;
  }

  // 2. 高度聚类
  cluster_heights();

  // 3. 检查是否有三个有效层
  int valid_layers = 0;
  for (int i = 0; i < 3; i++) {
    if (model_.layer_valid[i]) valid_layers++;
  }

  if (valid_layers < 3) {
    return false;
  }

  // 4. 检查层间距
  double gap1 = model_.layer_z[1] - model_.layer_z[0];
  double gap2 = model_.layer_z[2] - model_.layer_z[1];

  if (gap1 < layer_gap_min_ || gap1 > layer_gap_max_ || gap2 < layer_gap_min_ ||
      gap2 > layer_gap_max_) {
    tools::logger()->debug(
      "[OutpostTarget] Invalid layer gaps: {:.3f}, {:.3f}", gap1, gap2);
    return false;
  }

  // 5. 估算角速度
  if (observations_.size() >= 2) {
    // 用相邻观测估算角速度
    std::vector<double> omega_samples;
    for (size_t i = 1; i < observations_.size(); i++) {
      double dt = tools::delta_time(observations_[i].t, observations_[i - 1].t);
      if (dt > 0.001 && dt < 0.5) {
        double dyaw = tools::limit_rad(observations_[i].armor_yaw - observations_[i - 1].armor_yaw);
        omega_samples.push_back(dyaw / dt);
      }
    }

    if (!omega_samples.empty()) {
      // 取中位数作为角速度估计
      std::sort(omega_samples.begin(), omega_samples.end());
      model_.omega = omega_samples[omega_samples.size() / 2];

      // 前哨站转速约为 2.5 rad/s，限制范围
      model_.omega = std::clamp(model_.omega, -3.5, 3.5);
    }
  }

  // 6. 设置初始相位
  if (!observations_.empty()) {
    model_.phase_at_t0 = observations_.back().armor_yaw;
    model_.t0 = observations_.back().t;
  }

  model_.is_ready = true;
  return true;
}

void OutpostTarget::cluster_heights()
{
  if (observations_.empty()) return;

  // 提取所有相对高度
  std::vector<double> z_values;
  for (const auto & obs : observations_) {
    z_values.push_back(obs.z_relative);
  }

  // 排序
  std::sort(z_values.begin(), z_values.end());

  // 使用间隙检测进行聚类
  std::vector<std::vector<double>> clusters;
  clusters.push_back({z_values[0]});

  for (size_t i = 1; i < z_values.size(); i++) {
    double gap = z_values[i] - z_values[i - 1];
    if (gap > cluster_threshold_) {
      // 新的簇
      clusters.push_back({z_values[i]});
    } else {
      clusters.back().push_back(z_values[i]);
    }
  }

  // 如果簇太多，合并最近的簇
  while (clusters.size() > 3) {
    double min_gap = 1e10;
    size_t merge_idx = 0;
    for (size_t i = 0; i < clusters.size() - 1; i++) {
      double c1_max = *std::max_element(clusters[i].begin(), clusters[i].end());
      double c2_min = *std::min_element(clusters[i + 1].begin(), clusters[i + 1].end());
      double gap = c2_min - c1_max;
      if (gap < min_gap) {
        min_gap = gap;
        merge_idx = i;
      }
    }
    // 合并
    clusters[merge_idx].insert(
      clusters[merge_idx].end(), clusters[merge_idx + 1].begin(), clusters[merge_idx + 1].end());
    clusters.erase(clusters.begin() + merge_idx + 1);
  }

  // 如果簇太少，无法建模
  if (clusters.size() < 3) {
    // 可能只观测到了部分层，标记已有的
    for (size_t i = 0; i < clusters.size() && i < 3; i++) {
      double sum = std::accumulate(clusters[i].begin(), clusters[i].end(), 0.0);
      model_.layer_z[i] = sum / clusters[i].size();
      model_.layer_valid[i] = clusters[i].size() >= 2;  // 至少2个观测才认为有效
    }
    return;
  }

  // 正好3个簇
  for (size_t i = 0; i < 3; i++) {
    double sum = std::accumulate(clusters[i].begin(), clusters[i].end(), 0.0);
    model_.layer_z[i] = sum / clusters[i].size();
    model_.layer_valid[i] = clusters[i].size() >= 2;
  }
}

double OutpostTarget::calculate_angle_coverage() const
{
  if (observations_.size() < 2) return 0;

  // 收集所有角度
  std::vector<double> angles;
  for (const auto & obs : observations_) {
    angles.push_back(obs.armor_yaw);
  }

  // 找最大角度差
  double max_coverage = 0;
  for (size_t i = 0; i < angles.size(); i++) {
    for (size_t j = i + 1; j < angles.size(); j++) {
      double diff = std::abs(tools::limit_rad(angles[i] - angles[j]));
      max_coverage = std::max(max_coverage, diff);
    }
  }

  return max_coverage;
}

bool OutpostTarget::validate_observation(
  const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (!model_.is_ready) return false;

  // 1. 预测当前相位
  double dt = tools::delta_time(t, model_.t0);
  double predicted_phase = tools::limit_rad(model_.phase_at_t0 + model_.omega * dt);

  // 2. 识别观测到的是哪一层
  int layer = identify_layer(armor.xyz_in_world[2]);
  if (layer < 0) {
    tools::logger()->debug("[OutpostTarget] Cannot identify layer");
    return false;
  }

  // 3. 计算该层装甲板的预测位置
  // 假设三层装甲板相位相同 (简化模型)
  double pred_x = model_.center_x - model_.radius * std::cos(predicted_phase);
  double pred_y = model_.center_y - model_.radius * std::sin(predicted_phase);

  // 4. 计算xy平面误差
  double error = std::sqrt(
    std::pow(armor.xyz_in_world[0] - pred_x, 2) + std::pow(armor.xyz_in_world[1] - pred_y, 2));

  // 5. 检查误差是否在阈值内
  bool valid = error < validation_error_threshold_;

  if (valid) {
    // 更新模型相位
    model_.phase_at_t0 = armor.ypr_in_world[0];
    model_.t0 = t;
  }

  tools::logger()->debug(
    "[OutpostTarget] Validation: layer={}, error={:.3f}m, valid={}", layer, error, valid);

  return valid;
}

int OutpostTarget::identify_layer(double z) const
{
  double z_rel = z - model_.z_ref;

  double min_dist = 1e10;
  int best_layer = -1;

  for (int i = 0; i < 3; i++) {
    if (!model_.layer_valid[i]) continue;

    double dist = std::abs(z_rel - model_.layer_z[i]);
    if (dist < min_dist) {
      min_dist = dist;
      best_layer = i;
    }
  }

  // 如果距离太远，认为无效
  if (min_dist > layer_gap_max_ / 2) {
    return -1;
  }

  return best_layer;
}

void OutpostTarget::update_tracking(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  int layer = identify_layer(armor.xyz_in_world[2]);
  if (layer < 0) {
    consecutive_validation_failures_++;
    if (consecutive_validation_failures_ > 10) {
      tools::logger()->warn("[OutpostTarget] Too many failures, re-scanning");
      state_ = OutpostState::SCANNING;
      scan_start_time_ = t;
      observations_.clear();
      model_.is_ready = false;
    }
    return;
  }

  consecutive_validation_failures_ = 0;
  jumped = true;
  last_id = layer;

  // 更新EKF
  update_ekf(armor, layer);

  // 从EKF状态更新模型
  model_.center_x = ekf_.x[0];
  model_.center_y = ekf_.x[1];
  model_.z_ref = ekf_.x[2];
  model_.phase_at_t0 = ekf_.x[3];
  model_.omega = ekf_.x[4];
  model_.t0 = t;
}

void OutpostTarget::init_ekf(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  // 状态: [center_x, center_y, z_ref, phase, omega]
  Eigen::VectorXd x0(5);
  x0 << model_.center_x, model_.center_y, model_.z_ref, model_.phase_at_t0, model_.omega;

  Eigen::VectorXd P0_dig(5);
  P0_dig << 0.1, 0.1, 0.1, 0.5, 0.1;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;
  last_update_time_ = t;
}

void OutpostTarget::update_ekf(const Armor & armor, int layer_id)
{
  if (!ekf_initialized_) return;

  // 观测量: [armor_x, armor_y, armor_yaw]
  Eigen::Vector3d z;
  z << armor.xyz_in_world[0], armor.xyz_in_world[1], armor.ypr_in_world[0];

  // 观测函数 h(x)
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    double phase = x[3];
    double cx = x[0];
    double cy = x[1];
    double r = model_.radius;

    double armor_x = cx - r * std::cos(phase);
    double armor_y = cy - r * std::sin(phase);
    double armor_yaw = phase;  // 装甲板朝向等于相位

    return Eigen::Vector3d(armor_x, armor_y, armor_yaw);
  };

  // 观测雅可比矩阵 H
  double phase = ekf_.x[3];
  double r = model_.radius;

  Eigen::MatrixXd H(3, 5);
  // clang-format off
  H << 1, 0, 0, r * std::sin(phase), 0,   // d(armor_x)/dx
       0, 1, 0, -r * std::cos(phase), 0,  // d(armor_y)/dx
       0, 0, 0, 1, 0;                      // d(armor_yaw)/dx
  // clang-format on

  // 观测噪声
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
  R(0, 0) = 0.01;  // x方差
  R(1, 1) = 0.01;  // y方差
  R(2, 2) = 0.1;   // yaw方差

  // 减法函数 (处理角度回绕)
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[2] = tools::limit_rad(c[2]);
    return c;
  };

  ekf_.update(z, H, R, h, z_subtract);
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!model_.is_ready && state_ != OutpostState::TRACKING) {
    return list;
  }

  double phase = ekf_initialized_ ? ekf_.x[3] : model_.phase_at_t0;
  double cx = ekf_initialized_ ? ekf_.x[0] : model_.center_x;
  double cy = ekf_initialized_ ? ekf_.x[1] : model_.center_y;
  double z_base = ekf_initialized_ ? ekf_.x[2] : model_.z_ref;

  // 三个装甲板，相位差120度
  for (int i = 0; i < 3; i++) {
    double armor_phase = tools::limit_rad(phase + i * 2 * CV_PI / 3);
    double armor_x = cx - model_.radius * std::cos(armor_phase);
    double armor_y = cy - model_.radius * std::sin(armor_phase);
    double armor_z = z_base + model_.layer_z[i];

    list.push_back({armor_x, armor_y, armor_z, armor_phase});
  }

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  if (!ekf_initialized_) {
    // 返回兼容格式的状态向量
    // x vx y vy z vz a w r l h
    Eigen::VectorXd x(11);
    x << model_.center_x, 0, model_.center_y, 0, model_.z_ref, 0, model_.phase_at_t0, model_.omega,
      model_.radius, 0, 0;
    return x;
  }

  // 转换为兼容格式
  Eigen::VectorXd x(11);
  x << ekf_.x[0], 0, ekf_.x[1], 0, ekf_.x[2], 0, ekf_.x[3], ekf_.x[4], model_.radius, 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  // 检查角速度是否合理
  double omega = std::abs(ekf_.x[4]);
  if (omega > 5.0 || omega < 0.1) {
    return true;
  }

  return false;
}

double OutpostTarget::predict_phase(std::chrono::steady_clock::time_point t) const
{
  if (ekf_initialized_) {
    double dt = tools::delta_time(t, last_update_time_);
    return tools::limit_rad(ekf_.x[3] + ekf_.x[4] * dt);
  }
  double dt = tools::delta_time(t, model_.t0);
  return tools::limit_rad(model_.phase_at_t0 + model_.omega * dt);
}

Eigen::Vector3d OutpostTarget::predict_armor_position(double phase, int layer) const
{
  double cx = ekf_initialized_ ? ekf_.x[0] : model_.center_x;
  double cy = ekf_initialized_ ? ekf_.x[1] : model_.center_y;
  double z_base = ekf_initialized_ ? ekf_.x[2] : model_.z_ref;

  double armor_x = cx - model_.radius * std::cos(phase);
  double armor_y = cy - model_.radius * std::sin(phase);
  double armor_z = z_base + model_.layer_z[layer];

  return {armor_x, armor_y, armor_z};
}

}  // namespace auto_aim
