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
    layer_phase_offset_[i] = 0;
  }
}

void OutpostTarget::init_ekf(const Armor & armor, int layer)
{
  double armor_yaw = armor.ypr_in_world[0];
  double armor_x = armor.xyz_in_world[0];
  double armor_y = armor.xyz_in_world[1];
  double armor_z = armor.xyz_in_world[2];

  // 从装甲板位置推算旋转中心
  double cx = armor_x + outpost_radius_ * std::cos(armor_yaw);
  double cy = armor_y + outpost_radius_ * std::sin(armor_yaw);

  // 第一层的phase_offset = 0，phase0 = armor_yaw
  double phase0 = armor_yaw;

  // 状态: [cx, vx, cy, vy, phase, omega, radius]
  Eigen::VectorXd x0(7);
  x0 << cx, 0, cy, 0, phase0, 0, outpost_radius_;

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

  // 初始化该层
  layer_z_[layer] = armor_z;
  layer_initialized_[layer] = true;
  layer_observation_count_[layer] = 1;
  layer_phase_offset_[layer] = 0;  // 第一层的偏移为0

  tools::logger()->info(
    "[OutpostTarget] Init EKF from layer {}, cx={:.3f}, cy={:.3f}, phase0={:.3f}, z={:.3f}",
    layer, cx, cy, phase0, armor_z);
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

void OutpostTarget::update_layer(const Armor & armor, int layer)
{
  double armor_yaw = armor.ypr_in_world[0];

  // 如果是该层第一次更新，计算phase_offset
  if (layer_observation_count_[layer] == 0) {
    // 计算该装甲板相对于phase0的相位差
    double delta_phase = tools::limit_rad(armor_yaw - ekf_.x[4]);

    // 约束到最接近的 0°, 120°, 240° (排除已使用的)
    std::vector<double> candidates = {0, 2 * M_PI / 3, -2 * M_PI / 3};

    // 移除已被其他层使用的偏移
    for (int i = 0; i < 3; i++) {
      if (i != layer && layer_initialized_[i] && layer_observation_count_[i] > 0) {
        for (auto it = candidates.begin(); it != candidates.end();) {
          if (std::abs(tools::limit_rad(*it - layer_phase_offset_[i])) < 0.5) {
            it = candidates.erase(it);
          } else {
            ++it;
          }
        }
      }
    }

    // 选择最接近delta_phase的候选
    double best_offset = 0;
    double min_diff = 1e10;
    for (double cand : candidates) {
      double diff = std::abs(tools::limit_rad(delta_phase - cand));
      if (diff < min_diff) {
        min_diff = diff;
        best_offset = cand;
      }
    }

    layer_phase_offset_[layer] = best_offset;
    tools::logger()->info(
      "[OutpostTarget] Layer {} phase_offset={:.1f}° (observed delta={:.1f}°)",
      layer, best_offset * 57.3, delta_phase * 57.3);
  }

  // 更新该层的z (滑动平均)
  double alpha = 0.1;
  layer_z_[layer] = layer_z_[layer] * (1 - alpha) + armor.xyz_in_world[2] * alpha;
  layer_observation_count_[layer]++;

  // 该层的理论phase = phase0 + layer_phase_offset
  double phase_layer = tools::limit_rad(ekf_.x[4] + layer_phase_offset_[layer]);

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 计算雅可比矩阵
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];
  double phase_offset = layer_phase_offset_[layer];
  double phase_l = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_l);
  double armor_y = cy - r * std::sin(phase_l);
  Eigen::Vector3d armor_xyz(armor_x, armor_y, layer_z_[layer]);

  // d(armor_xyz) / d(state)
  // armor_x = cx - r*cos(phase0 + offset)
  // d(armor_x)/d(cx) = 1
  // d(armor_x)/d(phase0) = r*sin(phase0 + offset)
  // d(armor_x)/d(r) = -cos(phase0 + offset)
  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 7);
  H_xyz_state <<
    1, 0, 0, 0,  r * std::sin(phase_l), 0, -std::cos(phase_l),
    0, 0, 1, 0, -r * std::cos(phase_l), 0, -std::sin(phase_l),
    0, 0, 0, 0,                      0, 0,                  0;
  // clang-format on

  // d(ypd) / d(xyz)
  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // d(ypda) / d(state)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 7);
  H.block<3, 7>(0, 0) = H_ypd_xyz * H_xyz_state;
  H(3, 4) = 1;  // d(armor_yaw) / d(phase0) = 1

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

  // h函数
  auto h_with_z = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    double cx_ = x[0], cy_ = x[2];
    double phase0_ = x[4], r_ = x[6];
    double phase_l_ = tools::limit_rad(phase0_ + phase_offset);

    double ax = cx_ - r_ * std::cos(phase_l_);
    double ay = cy_ - r_ * std::sin(phase_l_);

    Eigen::Vector3d xyz(ax, ay, layer_z_[layer]);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase_l_};
  };

  ekf_.update(z_obs, H, R, h_with_z, z_subtract);
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
    // 第一次识别，初始化
    int layer = identify_layer(z);
    init_ekf(armor, layer);
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
    tools::logger()->debug("[OutpostTarget] Layer switch {} -> {}", current_layer_, layer);
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

  // 状态转移矩阵 [cx, vx, cy, vy, phase, omega, radius]
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

  // 过程噪声
  double v1 = 10;   // 位置加速度方差
  double v2 = 0.1;  // 角加速度方差
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

  // 前哨站转速限制
  if (update_count_ > 10 && std::abs(ekf_.x[5]) > 2) {
    ekf_.x[5] = ekf_.x[5] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int layer) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  // 使用该层学习到的相位偏移
  double phase_layer = tools::limit_rad(phase0 + layer_phase_offset_[layer]);

  double armor_x = cx - r * std::cos(phase_layer);
  double armor_y = cy - r * std::sin(phase_layer);
  double armor_z = layer_initialized_[layer] ? layer_z_[layer] : 0;

  return {armor_x, armor_y, armor_z, phase_layer};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  // 只返回有观测的层（这些层才有正确的phase_offset）
  for (int i = 0; i < 3; i++) {
    if (layer_initialized_[i] && layer_observation_count_[i] > 0) {
      list.push_back(armor_xyza(i));
    }
  }

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  // 转换为11维兼容格式: [cx, vx, cy, vy, z, vz, phase, omega, r, 0, 0]
  Eigen::VectorXd x(11);

  if (!ekf_initialized_) {
    x << 0, 0, 0, 0, 0, 0, 0, 0, outpost_radius_, 0, 0;
    return x;
  }

  // 取当前层的z，或第一个初始化层的z
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

  // ekf_.x = [cx, vx, cy, vy, phase, omega, radius]
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
