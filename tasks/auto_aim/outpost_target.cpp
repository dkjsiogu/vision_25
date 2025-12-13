#include "outpost_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

// ============== OutpostLayerEKF 实现 ==============

void OutpostLayerEKF::init(const Armor & armor, double radius)
{
  radius_ = radius;

  // 状态: [center_x, vx, center_y, vy, z, vz, phase, omega, radius]
  double armor_yaw = armor.ypr_in_world[0];

  // 从装甲板位置推算旋转中心
  double cx = armor.xyz_in_world[0] + radius_ * std::cos(armor_yaw);
  double cy = armor.xyz_in_world[1] + radius_ * std::sin(armor_yaw);
  double z = armor.xyz_in_world[2];

  Eigen::VectorXd x0(9);
  x0 << cx, 0, cy, 0, z, 0, armor_yaw, 0, radius_;

  Eigen::VectorXd P0_dig(9);
  P0_dig << 1, 64, 1, 64, 1, 64, 0.4, 1, 1e-4;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);  // phase
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  initialized_ = true;
  update_count_ = 1;
}

void OutpostLayerEKF::predict(double dt)
{
  if (!initialized_ || dt <= 0) return;

  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0, 0},  // center_x
    {0,  1,  0,  0,  0,  0,  0,  0, 0},  // vx
    {0,  0,  1, dt,  0,  0,  0,  0, 0},  // center_y
    {0,  0,  0,  1,  0,  0,  0,  0, 0},  // vy
    {0,  0,  0,  0,  1, dt,  0,  0, 0},  // z
    {0,  0,  0,  0,  0,  1,  0,  0, 0},  // vz
    {0,  0,  0,  0,  0,  0,  1, dt, 0},  // phase
    {0,  0,  0,  0,  0,  0,  0,  1, 0},  // omega
    {0,  0,  0,  0,  0,  0,  0,  0, 1}   // radius
  };
  // clang-format on

  // 过程噪声 - 前哨站位置稳定，转速稳定
  double v1 = 10;   // 位置加速度方差
  double v2 = 0.1;  // 角加速度方差
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速限制
  if (update_count_ > 10 && std::abs(ekf_.x[7]) > 2) {
    ekf_.x[7] = ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

void OutpostLayerEKF::update(const Armor & armor)
{
  if (!initialized_) return;

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z(4);
  z << ypd[0], ypd[1], ypd[2], ypr[0];

  // 观测函数 h(x) -> [yaw, pitch, distance, armor_yaw]
  auto h = [](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    double cx = x[0], cy = x[2], cz = x[4];
    double phase = x[6], r = x[8];

    double armor_x = cx - r * std::cos(phase);
    double armor_y = cy - r * std::sin(phase);

    Eigen::Vector3d xyz(armor_x, armor_y, cz);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase};
  };

  // 计算雅可比矩阵
  double cx = ekf_.x[0], cy = ekf_.x[2], cz = ekf_.x[4];
  double phase = ekf_.x[6], r = ekf_.x[8];

  double armor_x = cx - r * std::cos(phase);
  double armor_y = cy - r * std::sin(phase);
  Eigen::Vector3d armor_xyz(armor_x, armor_y, cz);

  // d(armor_xyz) / d(state)
  // clang-format off
  Eigen::MatrixXd H_xyz_state{
    {1, 0, 0, 0, 0, 0,  r * std::sin(phase), 0, -std::cos(phase)},  // d(armor_x)/d(state)
    {0, 0, 1, 0, 0, 0, -r * std::cos(phase), 0, -std::sin(phase)},  // d(armor_y)/d(state)
    {0, 0, 0, 0, 1, 0,                    0, 0,                 0}   // d(armor_z)/d(state)
  };
  // clang-format on

  // d(ypd) / d(xyz)
  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // d(ypda) / d(state)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 9);
  H.block<3, 9>(0, 0) = H_ypd_xyz * H_xyz_state;
  H(3, 6) = 1;  // d(armor_yaw) / d(phase) = 1

  // 观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
    log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 减法函数
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);  // yaw
    c[1] = tools::limit_rad(c[1]);  // pitch
    c[3] = tools::limit_rad(c[3]);  // armor_yaw
    return c;
  };

  ekf_.update(z, H, R, h, z_subtract);
  update_count_++;
}

void OutpostLayerEKF::apply_shared_constraint(
  double target_cx, double target_cy, double target_omega, double alpha)
{
  if (!initialized_) return;

  // 软约束: 将共享参数向目标值靠拢
  ekf_.x[0] = ekf_.x[0] * (1 - alpha) + target_cx * alpha;      // center_x
  ekf_.x[2] = ekf_.x[2] * (1 - alpha) + target_cy * alpha;      // center_y
  ekf_.x[7] = ekf_.x[7] * (1 - alpha) + target_omega * alpha;   // omega
}

Eigen::Vector4d OutpostLayerEKF::armor_xyza() const
{
  if (!initialized_) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2], cz = ekf_.x[4];
  double phase = ekf_.x[6], r = ekf_.x[8];

  double armor_x = cx - r * std::cos(phase);
  double armor_y = cy - r * std::sin(phase);

  return {armor_x, armor_y, cz, phase};
}

bool OutpostLayerEKF::diverged() const
{
  if (!initialized_) return true;

  double r = ekf_.x[8];
  if (r < 0.1 || r > 0.5) return true;

  double omega = std::abs(ekf_.x[7]);
  if (omega > 5.0) return true;

  return false;
}

// ============== OutpostTarget 实现 ==============

OutpostTarget::OutpostTarget(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

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
    case OutpostState::TRACKING:
      return "tracking";
    default:
      return "unknown";
  }
}

void OutpostTarget::reset()
{
  state_ = OutpostState::LOST;
  current_layer_ = -1;
  height_observations_.clear();
  temp_lost_count_ = 0;

  for (int i = 0; i < 3; i++) {
    layer_ekf_[i] = OutpostLayerEKF();
    layer_z_[i] = 0;
    layer_valid_[i] = false;
  }
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  last_update_time_ = t;
  temp_lost_count_ = 0;

  double z = armor.xyz_in_world[2];

  switch (state_) {
    case OutpostState::LOST: {
      state_ = OutpostState::SCANNING;
      scan_start_time_ = t;
      height_observations_.clear();
      z_ref_ = z;
      add_height_observation(z, t);
      tools::logger()->info("[OutpostTarget] Start scanning, z_ref={:.3f}", z_ref_);
      return false;
    }

    case OutpostState::SCANNING: {
      double elapsed = tools::delta_time(t, scan_start_time_);
      if (elapsed > scan_timeout_) {
        tools::logger()->warn("[OutpostTarget] Scan timeout ({:.1f}s), reset", elapsed);
        reset();
        return false;
      }

      add_height_observation(z, t);

      // 尝试聚类，即使没有完成三层也可以继续
      try_cluster_heights();

      // 收集足够观测后即可进入追踪（不需要等三层都识别到）
      if (height_observations_.size() >= static_cast<size_t>(min_observations_)) {
        // 尝试分配层，如果聚类完成就用聚类结果，否则动态分配
        int layer = identify_layer_dynamic(z);
        if (layer >= 0) {
          layer_ekf_[layer].init(armor, outpost_radius_);
          current_layer_ = layer;
          state_ = OutpostState::TRACKING;
          tools::logger()->info(
            "[OutpostTarget] Start tracking! layer={}, z={:.3f}, z_ref={:.3f}", layer, z, z_ref_);
          return true;
        }
      }
      return false;
    }

    case OutpostState::TRACKING: {
      int layer = identify_layer_dynamic(z);
      if (layer < 0) {
        tools::logger()->warn("[OutpostTarget] Cannot identify layer, z={:.3f}", z);
        return false;
      }

      if (layer != current_layer_) {
        jumped = true;
        tools::logger()->debug("[OutpostTarget] Switch layer {} -> {}", current_layer_, layer);
      }
      current_layer_ = layer;
      last_id = layer;

      // 如果该层EKF未初始化，初始化它
      if (!layer_ekf_[layer].is_initialized()) {
        layer_ekf_[layer].init(armor, outpost_radius_);
        tools::logger()->debug("[OutpostTarget] Init layer {} EKF", layer);
      } else {
        layer_ekf_[layer].update(armor);
      }

      // 应用共享约束
      apply_shared_constraints();

      return true;
    }

    default:
      return false;
  }
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  if (state_ != OutpostState::TRACKING) {
    return;
  }

  double dt = tools::delta_time(t, last_update_time_);
  if (dt <= 0 || dt > 0.1) return;

  predict(dt);
  last_update_time_ = t;
  temp_lost_count_++;

  if (temp_lost_count_ > max_temp_lost_count_) {
    tools::logger()->info("[OutpostTarget] Lost (temp_lost_count > {})", max_temp_lost_count_);
    reset();
  }
}

void OutpostTarget::predict(double dt)
{
  if (state_ != OutpostState::TRACKING) return;

  // 所有已初始化的层都进行预测
  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      layer_ekf_[i].predict(dt);
    }
  }

  // 预测后也应用共享约束
  apply_shared_constraints();
}

void OutpostTarget::add_height_observation(double z, std::chrono::steady_clock::time_point t)
{
  height_observations_.push_back({z, t});

  while (height_observations_.size() > 50) {
    height_observations_.pop_front();
  }
}

bool OutpostTarget::try_cluster_heights()
{
  if (height_observations_.size() < static_cast<size_t>(min_observations_)) {
    return false;
  }

  std::vector<double> z_values;
  for (const auto & obs : height_observations_) {
    z_values.push_back(obs.z - z_ref_);
  }

  std::sort(z_values.begin(), z_values.end());

  // 间隙聚类
  std::vector<std::vector<double>> clusters;
  clusters.push_back({z_values[0]});

  for (size_t i = 1; i < z_values.size(); i++) {
    double gap = z_values[i] - z_values[i - 1];
    if (gap > cluster_threshold_) {
      clusters.push_back({z_values[i]});
    } else {
      clusters.back().push_back(z_values[i]);
    }
  }

  // 合并过多的簇
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
    clusters[merge_idx].insert(
      clusters[merge_idx].end(), clusters[merge_idx + 1].begin(), clusters[merge_idx + 1].end());
    clusters.erase(clusters.begin() + merge_idx + 1);
  }

  if (clusters.size() < 3) {
    return false;
  }

  for (size_t i = 0; i < 3; i++) {
    double sum = std::accumulate(clusters[i].begin(), clusters[i].end(), 0.0);
    layer_z_[i] = sum / clusters[i].size();
    layer_valid_[i] = clusters[i].size() >= 2;
  }

  double gap1 = layer_z_[1] - layer_z_[0];
  double gap2 = layer_z_[2] - layer_z_[1];

  if (gap1 < layer_gap_min_ || gap1 > layer_gap_max_ || gap2 < layer_gap_min_ ||
      gap2 > layer_gap_max_) {
    tools::logger()->debug("[OutpostTarget] Invalid layer gaps: {:.3f}, {:.3f}", gap1, gap2);
    return false;
  }

  return layer_valid_[0] && layer_valid_[1] && layer_valid_[2];
}

int OutpostTarget::identify_layer(double z) const
{
  double z_rel = z - z_ref_;

  double min_dist = 1e10;
  int best_layer = -1;

  for (int i = 0; i < 3; i++) {
    if (!layer_valid_[i]) continue;

    double dist = std::abs(z_rel - layer_z_[i]);
    if (dist < min_dist) {
      min_dist = dist;
      best_layer = i;
    }
  }

  if (min_dist > layer_gap_max_ / 2) return -1;
  return best_layer;
}

int OutpostTarget::identify_layer_dynamic(double z)
{
  double z_rel = z - z_ref_;

  // 首先尝试用聚类结果
  int clustered_layer = identify_layer(z);
  if (clustered_layer >= 0) {
    return clustered_layer;
  }

  // 聚类未完成，使用动态分配
  // 检查已初始化的层，找到最接近的
  double min_dist = 1e10;
  int best_layer = -1;

  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      double layer_z = layer_ekf_[i].ekf().x[4] - z_ref_;  // 从EKF状态获取z
      double dist = std::abs(z_rel - layer_z);
      if (dist < min_dist) {
        min_dist = dist;
        best_layer = i;
      }
    }
  }

  // 如果找到接近的已初始化层（差距小于层间距的一半），使用它
  if (best_layer >= 0 && min_dist < layer_gap_min_ / 2) {
    return best_layer;
  }

  // 需要分配新层
  // 找一个未初始化的层
  int free_layer = -1;
  for (int i = 0; i < 3; i++) {
    if (!layer_ekf_[i].is_initialized()) {
      free_layer = i;
      break;
    }
  }

  if (free_layer >= 0) {
    // 记录这个层的相对高度
    layer_z_[free_layer] = z_rel;
    layer_valid_[free_layer] = true;
    tools::logger()->debug(
      "[OutpostTarget] Dynamic assign layer {}, z_rel={:.3f}", free_layer, z_rel);
    return free_layer;
  }

  // 所有层都已初始化但没找到匹配的，返回最近的
  return best_layer;
}

void OutpostTarget::apply_shared_constraints()
{
  // 计算已初始化层的共享参数加权平均
  double sum_cx = 0, sum_cy = 0, sum_omega = 0;
  double total_weight = 0;

  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      double weight = layer_ekf_[i].update_count();
      sum_cx += layer_ekf_[i].center_x() * weight;
      sum_cy += layer_ekf_[i].center_y() * weight;
      sum_omega += layer_ekf_[i].omega() * weight;
      total_weight += weight;
    }
  }

  if (total_weight < 1) return;

  double avg_cx = sum_cx / total_weight;
  double avg_cy = sum_cy / total_weight;
  double avg_omega = sum_omega / total_weight;

  // 应用软约束
  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      layer_ekf_[i].apply_shared_constraint(avg_cx, avg_cy, avg_omega, constraint_alpha_);
    }
  }
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (state_ != OutpostState::TRACKING) {
    return list;
  }

  // 返回三个装甲板的位置
  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      list.push_back(layer_ekf_[i].armor_xyza());
    }
  }

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  // 返回最近更新层的状态，转换为11维兼容格式
  int layer = current_layer_;
  if (layer < 0 || !layer_ekf_[layer].is_initialized()) {
    // 找一个已初始化的层
    for (int i = 0; i < 3; i++) {
      if (layer_ekf_[i].is_initialized()) {
        layer = i;
        break;
      }
    }
  }

  if (layer < 0 || !layer_ekf_[layer].is_initialized()) {
    Eigen::VectorXd x(11);
    x << 0, 0, 0, 0, 0, 0, 0, 0, outpost_radius_, 0, 0;
    return x;
  }

  // 转换: [cx,vx,cy,vy,z,vz,phase,omega,r] -> [cx,vx,cy,vy,z,vz,phase,omega,r,0,0]
  const auto & layer_x = layer_ekf_[layer].ekf().x;
  Eigen::VectorXd x(11);
  x << layer_x[0], layer_x[1], layer_x[2], layer_x[3], layer_x[4], layer_x[5], layer_x[6],
    layer_x[7], layer_x[8], 0, 0;
  return x;
}

const tools::ExtendedKalmanFilter & OutpostTarget::ekf() const
{
  static tools::ExtendedKalmanFilter dummy_ekf;

  int layer = current_layer_;
  if (layer >= 0 && layer_ekf_[layer].is_initialized()) {
    return layer_ekf_[layer].ekf();
  }

  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized()) {
      return layer_ekf_[i].ekf();
    }
  }

  return dummy_ekf;
}

bool OutpostTarget::diverged() const
{
  if (state_ != OutpostState::TRACKING) return false;

  // 检查当前层
  if (current_layer_ >= 0 && layer_ekf_[current_layer_].is_initialized()) {
    return layer_ekf_[current_layer_].diverged();
  }

  return false;
}

bool OutpostTarget::convergened() const
{
  if (state_ != OutpostState::TRACKING) return false;

  // 至少有一层收敛
  for (int i = 0; i < 3; i++) {
    if (layer_ekf_[i].is_initialized() && layer_ekf_[i].update_count() > 5) {
      return true;
    }
  }

  return false;
}

}  // namespace auto_aim
