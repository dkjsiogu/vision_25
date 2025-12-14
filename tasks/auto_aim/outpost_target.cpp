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
  current_id_ = -1;
  temp_lost_count_ = 0;
  jumped = false;
  last_id = 0;

  for (int i = 0; i < 3; i++) {
    height_offset_[i] = 0;
    height_offset_initialized_[i] = false;
  }
}

void OutpostTarget::init_ekf(const Armor & armor)
{
  double armor_yaw = armor.ypr_in_world[0];
  double armor_x = armor.xyz_in_world[0];
  double armor_y = armor.xyz_in_world[1];
  double armor_z = armor.xyz_in_world[2];

  // 从装甲板位置推算旋转中心
  double cx = armor_x + outpost_radius_ * std::cos(armor_yaw);
  double cy = armor_y + outpost_radius_ * std::sin(armor_yaw);

  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  // 初始化时，假设观测到的是id=0的装甲板，所以phase0 = armor_yaw
  Eigen::VectorXd x0(8);
  x0 << cx, 0, cy, 0, armor_z, armor_yaw, 0, outpost_radius_;

  // P0 参数与 sp_vision_25 保持一致
  Eigen::VectorXd P0_dig(8);
  P0_dig << 1, 64, 1, 64, 1, 0.4, 100, 1e-4;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[5] = tools::limit_rad(c[5]);  // phase0
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  // 初始化id=0的高度偏移为0（基准）
  height_offset_[0] = 0;
  height_offset_initialized_[0] = true;
  current_id_ = 0;

  // 调试：输出初始化信息
  tools::logger()->info(
    "[OutpostTarget] Init EKF: cx={:.3f}, cy={:.3f}, z={:.3f}, phase0={:.3f}, armor_xyz=({:.3f}, {:.3f}, {:.3f})",
    cx, cy, armor_z, armor_yaw, armor_x, armor_y, armor_z);
}

int OutpostTarget::match_armor_id(const Armor & armor) const
{
  if (!ekf_initialized_) return 0;

  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  double phase0 = ekf_.x[5];
  double cx = ekf_.x[0], cy = ekf_.x[2], z = ekf_.x[4], r = ekf_.x[7];

  double min_error = 1e10;
  int best_id = 0;

  // 使用与sp_vision_25相同的匹配逻辑：装甲板朝向角度差 + 观测方向角度差
  for (int id = 0; id < 3; id++) {
    double predicted_phase = tools::limit_rad(phase0 + id * 2 * M_PI / 3);

    // 预测的装甲板位置
    double pred_x = cx - r * std::cos(predicted_phase);
    double pred_y = cy - r * std::sin(predicted_phase);
    double pred_z = z + (height_offset_initialized_[id] ? height_offset_[id] : 0);

    // 预测的观测方向
    Eigen::Vector3d pred_xyz(pred_x, pred_y, pred_z);
    Eigen::Vector3d pred_ypd = tools::xyz2ypd(pred_xyz);

    // 角度误差 = 装甲板朝向角度差 + 观测yaw角度差
    double angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_phase)) +
                         std::abs(tools::limit_rad(armor.ypd_in_world[0] - pred_ypd[0]));

    if (angle_error < min_error) {
      min_error = angle_error;
      best_id = id;
    }
  }

  return best_id;
}

void OutpostTarget::update_armor(const Armor & armor, int id)
{
  // 更新该id的高度偏移 (相对于EKF中的z)
  // height_offset = observed_z - ekf_z
  double observed_z = armor.xyz_in_world[2];
  double ekf_z = ekf_.x[4];
  double new_offset = observed_z - ekf_z;

  if (!height_offset_initialized_[id]) {
    height_offset_[id] = new_offset;
    height_offset_initialized_[id] = true;
  } else {
    double alpha = 0.1;
    height_offset_[id] = height_offset_[id] * (1 - alpha) + new_offset * alpha;
  }

  // 该id的phase偏移
  double phase_offset = id * 2 * M_PI / 3;

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  double cx = ekf_.x[0], cy = ekf_.x[2], z = ekf_.x[4];
  double phase0 = ekf_.x[5], r = ekf_.x[7];
  double phase_id = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_id);
  double armor_y = cy - r * std::sin(phase_id);
  double armor_z = z + height_offset_[id];
  Eigen::Vector3d armor_xyz(armor_x, armor_y, armor_z);

  // H_xyz_state: 装甲板xyz对状态的雅可比 (3x8)
  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  // armor_x = cx - r * cos(phase0 + offset)
  // armor_y = cy - r * sin(phase0 + offset)
  // armor_z = z + height_offset (height_offset是常数)
  // d(armor_x)/d(cx) = 1, d(armor_x)/d(phase0) = r * sin(phase_id), d(armor_x)/d(r) = -cos(phase_id)
  // d(armor_y)/d(cy) = 1, d(armor_y)/d(phase0) = -r * cos(phase_id), d(armor_y)/d(r) = -sin(phase_id)
  // d(armor_z)/d(z) = 1
  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 8);
  H_xyz_state <<
    1, 0, 0, 0, 0,  r * std::sin(phase_id), 0, -std::cos(phase_id),
    0, 0, 1, 0, 0, -r * std::cos(phase_id), 0, -std::sin(phase_id),
    0, 0, 0, 0, 1,                       0, 0,                   0;
  // clang-format on

  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // 完整的观测雅可比 (4x8): [yaw, pitch, distance, armor_yaw] 对 [cx, vx, cy, vy, z, phase0, omega, r]
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 8);
  H.block<3, 8>(0, 0) = H_ypd_xyz * H_xyz_state;
  // d(armor_yaw)/d(phase0) = 1 (因为 armor_yaw = phase0 + offset)
  H(3, 5) = 1;

  // 观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
    std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  auto h_func = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    double cx_ = x[0], cy_ = x[2], z_ = x[4];
    double phase0_ = x[5], r_ = x[7];
    double phase_id_ = tools::limit_rad(phase0_ + phase_offset);

    double ax = cx_ - r_ * std::cos(phase_id_);
    double ay = cy_ - r_ * std::sin(phase_id_);
    double az = z_ + height_offset_[id];

    Eigen::Vector3d xyz(ax, ay, az);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase_id_};
  };

  ekf_.update(z_obs, H, R, h_func, z_subtract);
  update_count_++;
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  temp_lost_count_ = 0;
  priority = armor.priority;

  // 调试：输出观测到的装甲板位置
  tools::logger()->debug(
    "[OutpostTarget] Observed armor: xyz=({:.3f}, {:.3f}, {:.3f}), yaw={:.3f}",
    armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2], armor.ypr_in_world[0]);

  if (state_ == OutpostState::LOST) {
    // 第一次识别
    init_ekf(armor);
    current_id_ = 0;
    state_ = OutpostState::TRACKING;
    last_update_time_ = t;
    return true;
  }

  // 先 predict 到当前时刻，再 update（正确的 EKF 流程）
  double dt = tools::delta_time(t, last_update_time_);
  if (dt > 0 && dt < 0.1) {
    predict(dt);
  }
  last_update_time_ = t;

  // TRACKING状态: 用角度匹配确定id
  int id = match_armor_id(armor);

  if (id != current_id_) {
    jumped = true;
    tools::logger()->debug("[OutpostTarget] Jumped from id {} to id {}", current_id_, id);
  }
  current_id_ = id;
  last_id = id;

  update_armor(armor, id);

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

  // 状态转移矩阵: [cx, vx, cy, vy, z, phase0, omega, radius]
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0, 0},
    {0,  1,  0,  0,  0,  0,  0, 0},
    {0,  0,  1, dt,  0,  0,  0, 0},
    {0,  0,  0,  1,  0,  0,  0, 0},
    {0,  0,  0,  0,  1,  0,  0, 0},
    {0,  0,  0,  0,  0,  1, dt, 0},
    {0,  0,  0,  0,  0,  0,  1, 0},
    {0,  0,  0,  0,  0,  0,  0, 1}
  };
  // clang-format on

  double v1 = 10;   // 位置加速度方差
  double v2 = 0.1;  // 角加速度方差
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0, 0},
    {     0,      0,      0,      0,      0,      0,      0, 0},
    {     0,      0,      0,      0,      0, a * v2, b * v2, 0},
    {     0,      0,      0,      0,      0, b * v2, c * v2, 0},
    {     0,      0,      0,      0,      0,      0,      0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[5] = tools::limit_rad(x_prior[5]);  // phase0
    return x_prior;
  };

  // 限制角速度范围 (omega is at index 6)
  if (update_count_ > 10 && std::abs(ekf_.x[6]) > 2) {
    ekf_.x[6] = ekf_.x[6] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int id) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  double cx = ekf_.x[0], cy = ekf_.x[2], z = ekf_.x[4];
  double phase0 = ekf_.x[5], r = ekf_.x[7];

  double phase_id = tools::limit_rad(phase0 + id * 2 * M_PI / 3);

  double armor_x = cx - r * std::cos(phase_id);
  double armor_y = cy - r * std::sin(phase_id);

  // 使用EKF中的z + 该id的高度偏移
  double armor_z = z;
  if (height_offset_initialized_[id]) {
    armor_z += height_offset_[id];
  }

  return {armor_x, armor_y, armor_z, phase_id};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  // 返回所有已初始化高度偏移的装甲板
  for (int i = 0; i < 3; i++) {
    if (height_offset_initialized_[i]) {
      list.push_back(armor_xyza(i));
    }
  }

  // 如果只有一个装甲板被初始化，也输出它（用于初期追踪）
  if (list.empty() && ekf_initialized_) {
    list.push_back(armor_xyza(0));
  }

  // 调试日志
  if (!list.empty()) {
    tools::logger()->debug(
      "[OutpostTarget] armor_xyza_list: size={}, first=({:.3f}, {:.3f}, {:.3f})",
      list.size(), list[0][0], list[0][1], list[0][2]);
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

  // 8维状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  // 转换为Target兼容的11维状态: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3], ekf_.x[4], 0, ekf_.x[5], ekf_.x[6], ekf_.x[7], 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  // 状态: [cx, vx, cy, vy, z, phase0, omega, radius]
  double r = ekf_.x[7];
  if (r < 0.1 || r > 0.5) return true;

  double omega = std::abs(ekf_.x[6]);
  if (omega > 5.0) return true;

  return false;
}

bool OutpostTarget::convergened() const
{
  if (!ekf_initialized_) return false;
  return update_count_ > 10;
}

}  // namespace auto_aim
