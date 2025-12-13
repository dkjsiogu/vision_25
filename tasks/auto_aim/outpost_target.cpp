#include "outpost_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

OutpostTarget::OutpostTarget(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  if (yaml["outpost_max_temp_lost_count"]) {
    max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  }
  if (yaml["outpost_radius"]) {
    outpost_radius_ = yaml["outpost_radius"].as<double>();
  }
  if (yaml["outpost_z_cluster_threshold"]) {
    z_cluster_threshold_ = yaml["outpost_z_cluster_threshold"].as<double>();
  }

  reset();
}

std::string OutpostTarget::state_string() const
{
  switch (state_) {
    case OutpostState::LOST:
      return "lost";
    case OutpostState::TRACKING:
      return "tracking";
    default:
      return "unknown";
  }
}

void OutpostTarget::reset()
{
  state_ = OutpostState::LOST;
  ekf_initialized_ = false;
  update_count_ = 0;
  current_layer_ = -1;
  temp_lost_count_ = 0;
  jumped = false;
  last_id = 0;

  for (int i = 0; i < 3; i++) {
    layer_z_[i] = 0;
    layer_initialized_[i] = false;
    layer_observation_count_[i] = 0;
  }
}

void OutpostTarget::init_ekf(const Armor & armor)
{
  double armor_yaw = armor.ypr_in_world[0];
  double armor_x = armor.xyz_in_world[0];
  double armor_y = armor.xyz_in_world[1];

  // 从装甲板位置推算旋转中心
  double cx = armor_x + outpost_radius_ * std::cos(armor_yaw);
  double cy = armor_y + outpost_radius_ * std::sin(armor_yaw);

  // 状态: [cx, vx, cy, vy, phase, omega, radius]
  // phase是"层0装甲板"的相位
  Eigen::VectorXd x0(7);
  x0 << cx, 0, cy, 0, armor_yaw, 0, outpost_radius_;

  Eigen::VectorXd P0_dig(7);
  P0_dig << 0.5, 16, 0.5, 16, 0.4, 1, 1e-4;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[4] = tools::limit_rad(c[4]);  // phase
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  tools::logger()->info(
    "[OutpostTarget] Init EKF: cx={:.3f}, cy={:.3f}, phase={:.3f}", cx, cy, armor_yaw);
}

int OutpostTarget::identify_layer(double z)
{
  // 找最接近的已初始化层
  double min_dist = 1e10;
  int best_layer = -1;

  for (int i = 0; i < 3; i++) {
    if (layer_initialized_[i]) {
      double dist = std::abs(z - layer_z_[i]);
      if (dist < min_dist) {
        min_dist = dist;
        best_layer = i;
      }
    }
  }

  // 如果找到接近的层，返回它
  if (best_layer >= 0 && min_dist < z_cluster_threshold_) {
    return best_layer;
  }

  // 需要新建层，找一个未初始化的
  for (int i = 0; i < 3; i++) {
    if (!layer_initialized_[i]) {
      layer_z_[i] = z;
      layer_initialized_[i] = true;
      layer_observation_count_[i] = 0;
      tools::logger()->debug("[OutpostTarget] New layer {} at z={:.3f}", i, z);
      return i;
    }
  }

  // 所有层都初始化了，返回最近的
  return best_layer;
}

void OutpostTarget::reorder_layers_by_z()
{
  // 收集已初始化的层
  std::vector<std::pair<double, int>> z_layer_pairs;
  for (int i = 0; i < 3; i++) {
    if (layer_initialized_[i] && layer_observation_count_[i] > 0) {
      z_layer_pairs.push_back({layer_z_[i], i});
    }
  }

  if (z_layer_pairs.size() < 2) return;  // 不足2层，无需排序

  // 按z排序
  std::sort(z_layer_pairs.begin(), z_layer_pairs.end());

  // 检查是否需要重排
  bool need_reorder = false;
  for (size_t i = 0; i < z_layer_pairs.size(); i++) {
    if (z_layer_pairs[i].second != static_cast<int>(i)) {
      need_reorder = true;
      break;
    }
  }

  if (!need_reorder) return;

  // 重排：z最小的放层0，依此类推
  double new_layer_z[3] = {0, 0, 0};
  bool new_layer_initialized[3] = {false, false, false};
  int new_layer_observation_count[3] = {0, 0, 0};

  for (size_t i = 0; i < z_layer_pairs.size(); i++) {
    int old_layer = z_layer_pairs[i].second;
    new_layer_z[i] = layer_z_[old_layer];
    new_layer_initialized[i] = true;
    new_layer_observation_count[i] = layer_observation_count_[old_layer];
  }

  for (int i = 0; i < 3; i++) {
    layer_z_[i] = new_layer_z[i];
    layer_initialized_[i] = new_layer_initialized[i];
    layer_observation_count_[i] = new_layer_observation_count[i];
  }

  tools::logger()->debug(
    "[OutpostTarget] Reordered layers by z: [{:.3f}, {:.3f}, {:.3f}]",
    layer_z_[0], layer_z_[1], layer_z_[2]);
}

int OutpostTarget::get_sorted_layer_index(int layer) const
{
  // 返回该层按z排序后的索引（用于计算phase偏移）
  // 层0（z最小）→ 偏移0°，层1（z中）→ 偏移120°，层2（z最大）→ 偏移240°

  if (!layer_initialized_[layer]) return layer;

  int count = 0;
  for (int i = 0; i < 3; i++) {
    if (layer_initialized_[i] && layer_observation_count_[i] > 0 && layer_z_[i] < layer_z_[layer]) {
      count++;
    }
  }
  return count;
}

void OutpostTarget::update_layer(const Armor & armor, int layer)
{
  // 更新该层的z (滑动平均)
  double alpha = 0.1;
  layer_z_[layer] = layer_z_[layer] * (1 - alpha) + armor.xyz_in_world[2] * alpha;
  layer_observation_count_[layer]++;

  // 定期重排层（确保层号和z顺序一致）
  if (layer_observation_count_[layer] % 10 == 0) {
    reorder_layers_by_z();
  }

  // 该层的phase偏移（根据z排序：z最小→0°，z中→120°，z最大→240°）
  int sorted_idx = get_sorted_layer_index(layer);
  double phase_offset = sorted_idx * 2 * M_PI / 3;

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 计算雅可比矩阵
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];
  double phase_l = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_l);
  double armor_y = cy - r * std::sin(phase_l);
  Eigen::Vector3d armor_xyz(armor_x, armor_y, layer_z_[layer]);

  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 7);
  H_xyz_state <<
    1, 0, 0, 0,  r * std::sin(phase_l), 0, -std::cos(phase_l),
    0, 0, 1, 0, -r * std::cos(phase_l), 0, -std::sin(phase_l),
    0, 0, 0, 0,                      0, 0,                  0;
  // clang-format on

  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 7);
  H.block<3, 7>(0, 0) = H_ypd_xyz * H_xyz_state;
  H(3, 4) = 1;

  // 观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
    log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  auto h_func = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    double cx_ = x[0], cy_ = x[2];
    double phase0_ = x[4], r_ = x[6];
    double phase_l_ = tools::limit_rad(phase0_ + phase_offset);

    double ax = cx_ - r_ * std::cos(phase_l_);
    double ay = cy_ - r_ * std::sin(phase_l_);

    Eigen::Vector3d xyz(ax, ay, layer_z_[layer]);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase_l_};
  };

  ekf_.update(z_obs, H, R, h_func, z_subtract);
  update_count_++;
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  last_update_time_ = t;
  temp_lost_count_ = 0;
  priority = armor.priority;

  double z = armor.xyz_in_world[2];

  if (state_ == OutpostState::LOST) {
    // 第一次识别
    int layer = identify_layer(z);
    init_ekf(armor);
    layer_observation_count_[layer] = 1;
    current_layer_ = layer;
    state_ = OutpostState::TRACKING;
    return true;
  }

  // TRACKING状态
  int layer = identify_layer(z);
  if (layer < 0) {
    tools::logger()->warn("[OutpostTarget] Cannot identify layer, z={:.3f}", z);
    return false;
  }

  if (layer != current_layer_) {
    jumped = true;
  }
  current_layer_ = layer;
  last_id = layer;

  update_layer(armor, layer);

  return true;
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  if (state_ != OutpostState::TRACKING || !ekf_initialized_) {
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
  if (!ekf_initialized_ || dt <= 0) return;

  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0, 0},
    {0,  1,  0,  0,  0,  0, 0},
    {0,  0,  1, dt,  0,  0, 0},
    {0,  0,  0,  1,  0,  0, 0},
    {0,  0,  0,  0,  1, dt, 0},
    {0,  0,  0,  0,  0,  1, 0},
    {0,  0,  0,  0,  0,  0, 1}
  };
  // clang-format on

  double v1 = 10;
  double v2 = 0.1;
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0, 0},
    {b * v1, c * v1,      0,      0,      0,      0, 0},
    {     0,      0, a * v1, b * v1,      0,      0, 0},
    {     0,      0, b * v1, c * v1,      0,      0, 0},
    {     0,      0,      0,      0, a * v2, b * v2, 0},
    {     0,      0,      0,      0, b * v2, c * v2, 0},
    {     0,      0,      0,      0,      0,      0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[4] = tools::limit_rad(x_prior[4]);
    return x_prior;
  };

  if (update_count_ > 10 && std::abs(ekf_.x[5]) > 2) {
    ekf_.x[5] = ekf_.x[5] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int layer) const
{
  if (!ekf_initialized_ || !layer_initialized_[layer]) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  // 根据z排序确定phase偏移
  int sorted_idx = get_sorted_layer_index(layer);
  double phase_offset = sorted_idx * 2 * M_PI / 3;
  double phase_layer = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_layer);
  double armor_y = cy - r * std::sin(phase_layer);

  return {armor_x, armor_y, layer_z_[layer], phase_layer};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  for (int i = 0; i < 3; i++) {
    if (layer_initialized_[i] && layer_observation_count_[i] > 0) {
      list.push_back(armor_xyza(i));
    }
  }

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  Eigen::VectorXd x(11);

  if (!ekf_initialized_) {
    x << 0, 0, 0, 0, 0, 0, 0, 0, outpost_radius_, 0, 0;
    return x;
  }

  double z = 0;
  if (current_layer_ >= 0 && layer_initialized_[current_layer_]) {
    z = layer_z_[current_layer_];
  } else {
    for (int i = 0; i < 3; i++) {
      if (layer_initialized_[i]) {
        z = layer_z_[i];
        break;
      }
    }
  }

  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3], z, 0, ekf_.x[4], ekf_.x[5], ekf_.x[6], 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  double r = ekf_.x[6];
  if (r < 0.1 || r > 0.5) return true;

  double omega = std::abs(ekf_.x[5]);
  if (omega > 5.0) return true;

  return false;
}

bool OutpostTarget::convergened() const
{
  if (!ekf_initialized_) return false;
  return update_count_ > 10;
}

}  // namespace auto_aim
