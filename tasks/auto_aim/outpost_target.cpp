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
  if (yaml["pitch_stable_threshold"]) {
    pitch_stable_threshold_ = yaml["pitch_stable_threshold"].as<double>();
  }

  // 可选：pitch 不稳定时观测降权/高度滤波参数
  if (yaml["pitch_unstable_r_scale"]) {
    pitch_unstable_r_scale_ = yaml["pitch_unstable_r_scale"].as<double>();
  }
  if (yaml["observed_z_alpha_stable"]) {
    observed_z_alpha_stable_ = yaml["observed_z_alpha_stable"].as<double>();
  }
  if (yaml["observed_z_alpha_unstable"]) {
    observed_z_alpha_unstable_ = yaml["observed_z_alpha_unstable"].as<double>();
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
  temp_lost_count_ = 0;
  jumped = false;
  last_id = 0;

  observed_z_ = 0.0;
  observed_z_valid_ = false;

  pitch_history_.clear();
  pitch_variation_ = 1e10;
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

  // 初始化观测 z
  observed_z_ = armor_z;
  observed_z_valid_ = true;

  // 初始化 pitch 追踪
  double pitch = armor.ypd_in_world[1];
  pitch_history_.clear();
  pitch_history_.push_back(pitch);

  tools::logger()->info(
    "[OutpostTarget] Init: cx={:.3f}, cy={:.3f}, phase0={:.3f}, z={:.3f}",
    cx, cy, armor_yaw, armor_z);
}

void OutpostTarget::update_ekf(const Armor & armor)
{
  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  // 预测装甲板位置（使用 phase0，即装甲板 0 的角度）
  double armor_x = cx - r * std::cos(phase0);
  double armor_y = cy - r * std::sin(phase0);
  Eigen::Vector3d armor_xyz(armor_x, armor_y, observed_z_);

  // H_xyz_state: 装甲板xy对状态的雅可比 (3x7)
  // armor_x = cx - r * cos(phase0)
  // armor_y = cy - r * sin(phase0)
  // d(armor_x)/d(cx) = 1, d(armor_x)/d(phase0) = r * sin(phase0), d(armor_x)/d(r) = -cos(phase0)
  // d(armor_y)/d(cy) = 1, d(armor_y)/d(phase0) = -r * cos(phase0), d(armor_y)/d(r) = -sin(phase0)
  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 7);
  H_xyz_state <<
    1, 0, 0, 0,  r * std::sin(phase0), 0, -std::cos(phase0),
    0, 0, 1, 0, -r * std::cos(phase0), 0, -std::sin(phase0),
    0, 0, 0, 0,                     0, 0,                 0;  // z 不在状态中
  // clang-format on

  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // 完整的观测雅可比 (4x7): [yaw, pitch, distance, armor_yaw] 对状态
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 7);
  H.block<3, 7>(0, 0) = H_ypd_xyz * H_xyz_state;
  H(3, 4) = 1;  // d(armor_yaw)/d(phase0) = 1

  // 观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
    std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;

  // 高度切换/抖动期：降低 pitch 这维观测的影响，避免拖拽平面状态/相位
  if (!pitch_stable()) {
    R_dig[1] *= pitch_unstable_r_scale_;
  }
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

    double ax = cx_ - r_ * std::cos(phase0_);
    double ay = cy_ - r_ * std::sin(phase0_);
    double az = observed_z_;  // 直接用观测 z

    Eigen::Vector3d xyz(ax, ay, az);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase0_};
  };

  ekf_.update(z_obs, H, R, h_func, z_subtract);
  update_count_++;
}

void OutpostTarget::update_pitch_tracking(double pitch)
{
  pitch_history_.push_back(pitch);
  if (pitch_history_.size() > PITCH_HISTORY_SIZE) {
    pitch_history_.pop_front();
  }

  // 计算 pitch 变化幅度（最大值 - 最小值）
  if (pitch_history_.size() >= 3) {
    double min_pitch = *std::min_element(pitch_history_.begin(), pitch_history_.end());
    double max_pitch = *std::max_element(pitch_history_.begin(), pitch_history_.end());
    pitch_variation_ = max_pitch - min_pitch;
  }
}

bool OutpostTarget::pitch_stable() const
{
  return pitch_variation_ < pitch_stable_threshold_;
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  temp_lost_count_ = 0;
  priority = armor.priority;

  // 更新 pitch 追踪（用于判断是否处于高度切换/抖动期）
  update_pitch_tracking(armor.ypd_in_world[1]);

  // 更新观测 z（滑动平均）：稳定期快速跟随；不稳定期保守更新，避免多高度混入振荡。
  const double obs_z = armor.xyz_in_world[2];
  if (!observed_z_valid_) {
    observed_z_ = obs_z;
    observed_z_valid_ = true;
  } else {
    const double alpha = pitch_stable() ? observed_z_alpha_stable_ : observed_z_alpha_unstable_;
    observed_z_ = observed_z_ * (1.0 - alpha) + obs_z * alpha;
  }

  if (state_ == OutpostState::LOST) {
    init_ekf(armor);
    state_ = OutpostState::TRACKING;
    last_update_time_ = t;
    return true;
  }

  // 先 predict 到当前时刻
  double dt = tools::delta_time(t, last_update_time_);
  if (dt > 0 && dt < 0.1) {
    predict(dt);
  }
  last_update_time_ = t;

  // 更新 EKF
  update_ekf(armor);

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

  // 丢失一段时间后，omega 逐渐衰减
  if (temp_lost_count_ > 5) {
    ekf_.x[5] *= 0.92;
  }

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

  // 限制角速度范围
  if (update_count_ > 10 && std::abs(ekf_.x[5]) > 2.51) {
    ekf_.x[5] = ekf_.x[5] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int i) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  // 第 i 个装甲板的角度
  double angle = tools::limit_rad(phase0 + i * 2 * M_PI / 3);

  double armor_x = cx - r * std::cos(angle);
  double armor_y = cy - r * std::sin(angle);
  double armor_z = observed_z_;  // 所有装甲板用同一个观测 z

  return {armor_x, armor_y, armor_z, angle};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  // 返回三个装甲板
  for (int i = 0; i < 3; i++) {
    list.push_back(armor_xyza(i));
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

  // 7维状态: [cx, vx, cy, vy, phase0, omega, radius]
  // 转换为11维: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3],
       observed_z_, 0,  // z 用观测值，vz = 0
       ekf_.x[4], ekf_.x[5], ekf_.x[6], 0, 0;
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
