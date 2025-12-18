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
  if (yaml["outpost_z_gate"]) {
    outpost_z_gate_ = yaml["outpost_z_gate"].as<double>();
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
  current_zone_ = -1;
  temp_lost_count_ = 0;
  jumped = false;
  last_id = 0;

  for (int i = 0; i < 3; i++) {
    zone_z_[i] = 0;
    zone_z_initialized_[i] = false;
    last_zone_yaw_initialized_[i] = false;
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

  // 状态: [cx, vx, cy, vy, phase0, omega, radius] (7维)
  // phase0 = 第一个观测到的装甲板的朝向
  Eigen::VectorXd x0(7);
  x0 << cx, 0, cy, 0, armor_yaw, 0, outpost_radius_;

  // P0 参数
  Eigen::VectorXd P0_dig(7);
  P0_dig << 1, 64, 1, 64, 0.4, 100, 1e-4;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[4] = tools::limit_rad(c[4]);  // phase0
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  // 初始化 zone 0 的 z 值
  zone_z_[0] = armor_z;
  zone_z_initialized_[0] = true;
  current_zone_ = 0;

  tools::logger()->info(
    "[OutpostTarget] Init EKF: cx={:.3f}, cy={:.3f}, phase0={:.3f}, z={:.3f}",
    cx, cy, armor_yaw, armor_z);
}

int OutpostTarget::match_zone(const Armor & armor) const
{
  // 初始化阶段：只有一个观测，直接当 zone 0
  if (!ekf_initialized_) return 0;

  // 观测量
  const double obs_armor_yaw = armor.ypr_in_world[0];
  const Eigen::Vector3d obs_ypd = armor.ypd_in_world;
  const double obs_z = armor.xyz_in_world[2];

  const double phase0 = ekf_.x[4];
  const double cx = ekf_.x[0];
  const double cy = ekf_.x[2];
  const double r = ekf_.x[6];

  // 权重：armor_yaw 残差是最可靠的（与高度无关），yaw/pitch/dist 用于打破边界抖动
  constexpr double w_armor_yaw = 3.0;
  constexpr double w_yaw = 1.0;
  constexpr double w_pitch = 0.3;
  constexpr double w_dist = 0.1;

  int best_zone = 0;
  double best_cost = 1e100;

  for (int zone = 0; zone < 3; zone++) {
    const double phase_zone = tools::limit_rad(phase0 + zone * 2 * M_PI / 3);

    // 预测 xyz：z 若未初始化则暂用本次观测 z（否则 pitch/dist 会把 cost 弄乱）
    const double z_use = zone_z_initialized_[zone] ? zone_z_[zone] : obs_z;
    const double ax = cx - r * std::cos(phase_zone);
    const double ay = cy - r * std::sin(phase_zone);
    const Eigen::Vector3d pred_xyz(ax, ay, z_use);
    const Eigen::Vector3d pred_ypd = tools::xyz2ypd(pred_xyz);

    const double e_armor_yaw = std::abs(tools::limit_rad(obs_armor_yaw - phase_zone));
    const double e_yaw = std::abs(tools::limit_rad(obs_ypd[0] - pred_ypd[0]));
    const double e_pitch = std::abs(tools::limit_rad(obs_ypd[1] - pred_ypd[1]));
    const double e_dist = std::abs(obs_ypd[2] - pred_ypd[2]) / std::max(1.0, obs_ypd[2]);

    double cost = w_armor_yaw * e_armor_yaw + w_yaw * e_yaw + w_pitch * e_pitch + w_dist * e_dist;

    // 滞回：偏向维持当前 zone，减少边界处抖动（尤其是低->高那类突变附近）
    if (zone == current_zone_) cost -= 0.05;

    if (cost < best_cost) {
      best_cost = cost;
      best_zone = zone;
    }
  }

  return best_zone;
}

void OutpostTarget::update_omega_from_observation(
  int zone, double armor_yaw, std::chrono::steady_clock::time_point t)
{
  if (zone < 0 || zone >= 3 || !ekf_initialized_) return;

  if (!last_zone_yaw_initialized_[zone]) {
    last_zone_yaw_[zone] = armor_yaw;
    last_zone_time_[zone] = t;
    last_zone_yaw_initialized_[zone] = true;
    return;
  }

  const double dt = tools::delta_time(t, last_zone_time_[zone]);
  if (dt <= 0.0 || dt > 0.1) {
    last_zone_yaw_[zone] = armor_yaw;
    last_zone_time_[zone] = t;
    return;
  }

  const double dyaw = tools::limit_rad(armor_yaw - last_zone_yaw_[zone]);
  const double omega_meas = dyaw / dt;

  // 只在尚未稳定前（或 omega 很离谱时）用观测来拉一把；避免稳定后被噪声扰动
  const bool not_converged = update_count_ < 8;
  const bool omega_bad = std::abs(ekf_.x[5]) > 6.0;
  if (not_converged || omega_bad) {
    const double beta = not_converged ? 0.35 : 0.15;
    ekf_.x[5] = (1.0 - beta) * ekf_.x[5] + beta * omega_meas;
  }

  last_zone_yaw_[zone] = armor_yaw;
  last_zone_time_[zone] = t;
}

void OutpostTarget::update_zone(const Armor & armor, int zone)
{
  double observed_z = armor.xyz_in_world[2];

  // 更新该 zone 的 z 值
  if (!zone_z_initialized_[zone]) {
    zone_z_[zone] = observed_z;
    zone_z_initialized_[zone] = true;
  } else {
    // 高/中/低在切换时会突变，误关联一次会把该层高度污染很久。
    // 用门限拒绝明显不合理的 z 跳变（z_gate 可在 yaml 里配置 outpost_z_gate）。
    if (std::abs(observed_z - zone_z_[zone]) > outpost_z_gate_) {
      tools::logger()->debug(
        "[OutpostTarget] Reject z update: zone={}, z_obs={:.3f}, z_hist={:.3f}", zone, observed_z,
        zone_z_[zone]);
    } else {
    double alpha = 0.1;
    zone_z_[zone] = zone_z_[zone] * (1 - alpha) + observed_z * alpha;
    }
  }

  // 该 zone 的 phase 偏移
  double phase_offset = zone * 2 * M_PI / 3;

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];
  double phase_zone = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_zone);
  double armor_y = cy - r * std::sin(phase_zone);
  double armor_z = zone_z_[zone];
  Eigen::Vector3d armor_xyz(armor_x, armor_y, armor_z);

  // H_xyz_state: 装甲板xy对状态的雅可比 (2x7)
  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
  // armor_x = cx - r * cos(phase0 + offset)
  // armor_y = cy - r * sin(phase0 + offset)
  // d(armor_x)/d(cx) = 1, d(armor_x)/d(phase0) = r * sin(phase_zone), d(armor_x)/d(r) = -cos(phase_zone)
  // d(armor_y)/d(cy) = 1, d(armor_y)/d(phase0) = -r * cos(phase_zone), d(armor_y)/d(r) = -sin(phase_zone)

  // 完整的 xyz 对状态的雅可比 (3x7)，z 行全为 0（因为 z 不在 EKF 状态中）
  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 7);
  H_xyz_state <<
    1, 0, 0, 0,  r * std::sin(phase_zone), 0, -std::cos(phase_zone),
    0, 0, 1, 0, -r * std::cos(phase_zone), 0, -std::sin(phase_zone),
    0, 0, 0, 0,                         0, 0,                     0;
  // clang-format on

  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // 完整的观测雅可比 (4x7): [yaw, pitch, distance, armor_yaw] 对 [cx, vx, cy, vy, phase0, omega, r]
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 7);
  H.block<3, 7>(0, 0) = H_ypd_xyz * H_xyz_state;
  // d(armor_yaw)/d(phase0) = 1 (因为 armor_yaw = phase0 + offset)
  H(3, 4) = 1;

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
    double cx_ = x[0], cy_ = x[2];
    double phase0_ = x[4], r_ = x[6];
    double phase_zone_ = tools::limit_rad(phase0_ + phase_offset);

    double ax = cx_ - r_ * std::cos(phase_zone_);
    double ay = cy_ - r_ * std::sin(phase_zone_);
    double az = zone_z_[zone];  // z 直接用记录的值

    Eigen::Vector3d xyz(ax, ay, az);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase_zone_};
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

  tools::logger()->debug(
    "[OutpostTarget] Observed armor: xyz=({:.3f}, {:.3f}, {:.3f}), yaw={:.3f}",
    armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2], armor.ypr_in_world[0]);

  if (state_ == OutpostState::LOST) {
    init_ekf(armor);
    current_zone_ = 0;
    state_ = OutpostState::TRACKING;
    last_update_time_ = t;
    return true;
  }

  // 先 predict 到当前时刻，再 update
  double dt = tools::delta_time(t, last_update_time_);
  if (dt > 0 && dt < 0.1) {
    predict(dt);
  }
  last_update_time_ = t;

  // 基于预测残差关联 zone，比固定相位区间更稳
  int zone = match_zone(armor);

  if (zone != current_zone_) {
    jumped = true;
    tools::logger()->debug("[OutpostTarget] Jumped from zone {} to zone {}", current_zone_, zone);
  }
  current_zone_ = zone;
  last_id = zone;

  // 用当前观测差分辅助 omega 初始/纠偏
  update_omega_from_observation(zone, armor.ypr_in_world[0], t);

  update_zone(armor, zone);

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

  // 状态转移矩阵: [cx, vx, cy, vy, phase0, omega, radius]
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
    x_prior[4] = tools::limit_rad(x_prior[4]);  // phase0
    return x_prior;
  };

  // 限制角速度范围 (omega is at index 5)
  if (update_count_ > 10 && std::abs(ekf_.x[5]) > 2.51) {
    ekf_.x[5] = ekf_.x[5] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int zone) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  double phase_zone = tools::limit_rad(phase0 + zone * 2 * M_PI / 3);

  double armor_x = cx - r * std::cos(phase_zone);
  double armor_y = cy - r * std::sin(phase_zone);

  // z 直接用该 zone 记录的值
  double armor_z = zone_z_initialized_[zone] ? zone_z_[zone] : zone_z_[current_zone_];

  return {armor_x, armor_y, armor_z, phase_zone};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  // 返回所有已初始化 z 的 zone
  for (int i = 0; i < 3; i++) {
    if (zone_z_initialized_[i]) {
      list.push_back(armor_xyza(i));
    }
  }

  // 如果没有任何 zone 初始化，返回当前 zone
  if (list.empty()) {
    list.push_back(armor_xyza(current_zone_));
  }

  tools::logger()->debug(
    "[OutpostTarget] armor_xyza_list: size={}, zones_initialized=[{},{},{}]",
    list.size(), zone_z_initialized_[0], zone_z_initialized_[1], zone_z_initialized_[2]);

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  Eigen::VectorXd x(11);

  if (!ekf_initialized_) {
    x << 0, 0, 0, 0, 0, 0, 0, 0, outpost_radius_, 0, 0;
    return x;
  }

  // 7维状态: [cx, vx, cy, vy, phase0, omega, radius]
  // 转换为Target兼容的11维状态: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  // z 用当前 zone 的值
  double z = zone_z_initialized_[current_zone_] ? zone_z_[current_zone_] : 0;
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3], z, 0, ekf_.x[4], ekf_.x[5], ekf_.x[6], 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
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
