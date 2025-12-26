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

  // 可选：omega PLL 参数
  if (yaml["outpost_omega_max_abs"]) {
    omega_max_abs_ = yaml["outpost_omega_max_abs"].as<double>();
  }
  if (yaml["outpost_pll_Kp"]) {
    pll_Kp_ = yaml["outpost_pll_Kp"].as<double>();
  }

  // 可选：(x,y) 观测噪声参数
  if (yaml["outpost_sigma_xy_base"]) {
    sigma_xy_base_ = yaml["outpost_sigma_xy_base"].as<double>();
  }
  if (yaml["outpost_sigma_xy_k"]) {
    sigma_xy_k_ = yaml["outpost_sigma_xy_k"].as<double>();
  }
  if (yaml["outpost_sigma_xy_min"]) {
    sigma_xy_min_ = yaml["outpost_sigma_xy_min"].as<double>();
  }
  if (yaml["outpost_sigma_xy_max"]) {
    sigma_xy_max_ = yaml["outpost_sigma_xy_max"].as<double>();
  }

  // 可选：(x,y) 残差门控
  if (yaml["outpost_xy_residual_gate_m"]) {
    xy_residual_gate_m_ = yaml["outpost_xy_residual_gate_m"].as<double>();
  }
  if (yaml["outpost_xy_residual_ratio_gate"]) {
    xy_residual_ratio_gate_ = yaml["outpost_xy_residual_ratio_gate"].as<double>();
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

  omega_est_ = 0.0;
  unwrapped_phase_history_.clear();
  phase_time_history_.clear();
  unwrapped_phase_accum_ = 0.0;

  phase0_pred_valid_ = false;
  last_pll_time_valid_ = false;

  meas_plate_id_ = 0;
  meas_valid_ = true;
  last_obs_phase_valid_ = false;
}

void OutpostTarget::align_phase_to_observation_xy(const Armor & armor)
{
  if (!ekf_initialized_) return;

  const double cx = ekf_.x[0], cy = ekf_.x[2];
  const double phase0_pred = ekf_.x[4];
  const double r = ekf_.x[5];

  const double obs_x = armor.xyz_in_world[0];
  const double obs_y = armor.xyz_in_world[1];

  // 观测可能来自三块装甲板中的任意一块：
  // 选取使 (x,y) 残差最小的那一块，记录对应 plate id。
  // 同时记录第二小的残差，用于比值门控。
  int best_i = 0;
  double best_err2 = 1e100;
  double second_err2 = 1e100;

  for (int i = 0; i < 3; i++) {
    const double angle = tools::limit_rad(phase0_pred + i * 2.0 * M_PI / 3.0);
    const double pred_x = cx - r * std::cos(angle);
    const double pred_y = cy - r * std::sin(angle);
    const double dx = obs_x - pred_x;
    const double dy = obs_y - pred_y;
    const double err2 = dx * dx + dy * dy;
    if (err2 < best_err2) {
      second_err2 = best_err2;
      best_err2 = err2;
      best_i = i;
    } else if (err2 < second_err2) {
      second_err2 = err2;
    }
  }

  meas_plate_id_ = best_i;

  // 双重门控：
  // 1. 绝对门控：最小残差不能太大
  // 2. 比值门控：最小残差必须显著优于第二名（避免中心漂移时三个都大但仍选出一个）
  const double abs_gate = xy_residual_gate_m_ * xy_residual_gate_m_;
  const double ratio = (second_err2 > 1e-6) ? (best_err2 / second_err2) : 0.0;
  meas_valid_ = (best_err2 <= abs_gate) && (ratio < xy_residual_ratio_gate_);

  if (best_i != 0) {
    jumped = true;
    last_id = best_i;
  }
}

void OutpostTarget::update_omega_from_observation_xy(
  const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (!ekf_initialized_) return;
  if (!meas_valid_) return;
  if (!phase0_pred_valid_) return;

  const double cx = ekf_.x[0], cy = ekf_.x[2];
  const double obs_x = armor.xyz_in_world[0];
  const double obs_y = armor.xyz_in_world[1];

  // 用中心->装甲板的几何关系反推出"观测相位"（不依赖 PnP 的装甲板朝向 yaw）。
  // armor_x = cx - r cos(phase) => (cx - armor_x, cy - armor_y) 与 (cos,sin) 同向
  const double obs_phase_plate = std::atan2(cy - obs_y, cx - obs_x);
  // 转换为"plate0"的相位观测，避免切板造成的 120° 跳变污染 omega
  const double obs_phase0 = tools::limit_rad(obs_phase_plate - meas_plate_id_ * 2.0 * M_PI / 3.0);

  // ========== PLL 部分：用相位误差修正 omega ==========
  // 使用“predict后、update前”的相位作为预测值，避免 update_ekf() 已经吃入观测导致误差被人为变小。
  const double pred_phase0 = phase0_pred_before_update_;
  const double phase_err = tools::limit_rad(obs_phase0 - pred_phase0);

  // dt 无关：把 pll_Kp_ 视为 omega_dot 的增益（1/s^2 量纲），离散实现 omega += Kp * err * dt
  double dt_pll = 0.0;
  if (last_pll_time_valid_) {
    dt_pll = tools::delta_time(t, last_pll_time_);
  }
  last_pll_time_ = t;
  last_pll_time_valid_ = true;

  if (dt_pll > 0.0 && dt_pll < 0.12) {
    // 前几帧用较大的 Kp 快速点火，之后用较小的 Kp 平滑跟踪
    const double Kp = (update_count_ < 5) ? (pll_Kp_ * 2.0) : pll_Kp_;
    omega_est_ += Kp * phase_err * dt_pll;
  }

  // ========== 滑窗回归部分：用展开相位序列拟合 omega ==========
  // 相位展开：累积相位跳变
  if (last_obs_phase_valid_) {
    const double dphase = tools::limit_rad(obs_phase0 - last_obs_phase_);
    // 跳变门控
    if (std::abs(dphase) < 1.2) {
      unwrapped_phase_accum_ += dphase;
    }
  }
  last_obs_phase_ = obs_phase0;
  last_obs_time_ = t;
  last_obs_phase_valid_ = true;

  // 记录到历史队列
  const double time_sec = std::chrono::duration<double>(t.time_since_epoch()).count();
  unwrapped_phase_history_.push_back(unwrapped_phase_accum_);
  phase_time_history_.push_back(time_sec);
  while (unwrapped_phase_history_.size() > OMEGA_WINDOW_SIZE) {
    unwrapped_phase_history_.pop_front();
    phase_time_history_.pop_front();
  }

  // 滑窗回归：拟合 omega = d(phase)/dt
  if (unwrapped_phase_history_.size() >= 5) {
    double t_mean = 0, phi_mean = 0;
    const size_t n = unwrapped_phase_history_.size();
    for (size_t i = 0; i < n; i++) {
      t_mean += phase_time_history_[i];
      phi_mean += unwrapped_phase_history_[i];
    }
    t_mean /= n;
    phi_mean /= n;

    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
      const double dt_i = phase_time_history_[i] - t_mean;
      const double dphi_i = unwrapped_phase_history_[i] - phi_mean;
      num += dt_i * dphi_i;
      den += dt_i * dt_i;
    }

    if (den > 1e-9) {
      const double omega_regress = num / den;
      // 融合 PLL 和回归结果：回归更稳定，权重逐渐增大
      const double regress_weight = std::min(1.0, update_count_ / 15.0);
      omega_est_ = (1.0 - regress_weight) * omega_est_ + regress_weight * omega_regress;
    }
  }

  // 限幅
  omega_est_ = std::clamp(omega_est_, -omega_max_abs_, omega_max_abs_);
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

  // 状态: [cx, vx, cy, vy, phase0, radius] (6维)
  Eigen::VectorXd x0(6);
  x0 << cx, 0, cy, 0, armor_yaw, outpost_radius_;

  // P0 参数
  Eigen::VectorXd P0_dig(6);
  P0_dig << 1, 64, 1, 64, 0.4, 0.01;  // radius 初始不确定性小
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

  // 初始化 plate-id / 相位差分缓存，帮助 omega 更快点火
  meas_plate_id_ = 0;
  meas_valid_ = true;
  last_obs_phase_ = tools::limit_rad(std::atan2(cy - armor_y, cx - armor_x));
  last_obs_time_ = std::chrono::steady_clock::time_point{};  // 由首次 update() 写入
  last_obs_phase_valid_ = false;  // 下一帧开始做差分

  omega_est_ = 0.0;
  unwrapped_phase_history_.clear();
  phase_time_history_.clear();
  unwrapped_phase_accum_ = 0.0;

  phase0_pred_valid_ = false;
  last_pll_time_valid_ = false;

  tools::logger()->info(
    "[OutpostTarget] Init: cx={:.3f}, cy={:.3f}, phase0={:.3f}, z={:.3f}",
    cx, cy, armor_yaw, armor_z);
}

void OutpostTarget::update_ekf(const Armor & armor)
{
  if (!meas_valid_) {
    return;
  }

  // 观测量：直接用世界系装甲板 (x,y)
  // 这样 phase/omega 由位置序列驱动，不依赖 PnP 的装甲板朝向 yaw（该量在前哨站场景下很可能不稳定/近似常量）。
  Eigen::VectorXd z_obs(2);
  z_obs << armor.xyz_in_world[0], armor.xyz_in_world[1];

  // 状态: [cx, vx, cy, vy, phase0, radius]
  // 当前帧观测对应的装甲板角度：phase0 + meas_plate_id*120deg
  const double phase0 = ekf_.x[4];
  const double r = ekf_.x[5];
  const double angle = tools::limit_rad(phase0 + meas_plate_id_ * 2.0 * M_PI / 3.0);

  // 观测雅可比 (2x6)
  // ax = cx - r cos(angle)
  // ay = cy - r sin(angle)
  // d(ax)/d(cx) = 1, d(ax)/d(phase0) = r*sin(angle), d(ax)/d(r) = -cos(angle)
  // d(ay)/d(cy) = 1, d(ay)/d(phase0) = -r*cos(angle), d(ay)/d(r) = -sin(angle)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 6);
  H(0, 0) = 1.0;
  H(0, 4) = r * std::sin(angle);
  H(0, 5) = -std::cos(angle);
  H(1, 2) = 1.0;
  H(1, 4) = -r * std::cos(angle);
  H(1, 5) = -std::sin(angle);

  // 观测噪声：用距离粗略缩放。pitch 不稳定时整体加大 (x,y) 噪声，避免高度抖动导致的解算漂移拖拽中心/相位。
  const double dist = std::max(0.0, armor.ypd_in_world[2]);
  double sigma_xy = sigma_xy_base_ + sigma_xy_k_ * dist;
  sigma_xy = std::clamp(sigma_xy, sigma_xy_min_, sigma_xy_max_);
  if (!pitch_stable()) {
    sigma_xy *= std::sqrt(pitch_unstable_r_scale_);
  }
  Eigen::VectorXd R_dig(2);
  R_dig << sigma_xy * sigma_xy, sigma_xy * sigma_xy;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    return a - b;
  };

  auto h_func = [&](const Eigen::VectorXd & x) -> Eigen::Vector2d {
    const double cx_ = x[0], cy_ = x[2];
    const double phase0_ = x[4], r_ = x[5];
    const double angle_ = tools::limit_rad(phase0_ + meas_plate_id_ * 2.0 * M_PI / 3.0);
    const double ax = cx_ - r_ * std::cos(angle_);
    const double ay = cy_ - r_ * std::sin(angle_);
    return {ax, ay};
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

  // 观测可能来自不同装甲板：更新前基于 (x,y) 残差对齐 phase0，保持残差连续
  align_phase_to_observation_xy(armor);

  // 缓存本帧 update 前的预测相位，供 PLL 使用
  phase0_pred_before_update_ = ekf_.x[4];
  phase0_pred_valid_ = true;

  // 更新 EKF
  update_ekf(armor);

  // 用几何相位差分辅助 omega 估计（带跳变门控）
  update_omega_from_observation_xy(armor, t);

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
    omega_est_ *= 0.92;
  }

  if (temp_lost_count_ > max_temp_lost_count_) {
    tools::logger()->info("[OutpostTarget] Lost (temp_lost_count > {})", max_temp_lost_count_);
    reset();
  }
}

void OutpostTarget::predict(double dt)
{
  if (!ekf_initialized_ || dt <= 0) return;

  // 状态转移矩阵: [cx, vx, cy, vy, phase0, radius]
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0, 0},
    {0,  1,  0,  0,  0, 0},
    {0,  0,  1, dt,  0, 0},
    {0,  0,  0,  1,  0, 0},
    {0,  0,  0,  0,  1, 0},
    {0,  0,  0,  0,  0, 1}
  };
  // clang-format on

  double v1 = 10;      // 位置加速度方差
  double vphi = 0.2;   // 相位随机游走强度（吸收 omega 估计误差）
  double vr = 1e-6;    // 半径过程噪声（很小，能缓慢自适应但不会乱跳）
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,        0,    0},
    {b * v1, c * v1,      0,      0,        0,    0},
    {     0,      0, a * v1, b * v1,        0,    0},
    {     0,      0, b * v1, c * v1,        0,    0},
    {     0,      0,      0,      0, c * vphi,    0},
    {     0,      0,      0,      0,        0, c * vr}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    // phase0 由 omega_est_ 外推
    x_prior[4] = tools::limit_rad(x_prior[4] + omega_est_ * dt);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int i) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4];
  double r = ekf_.x[5];

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

  // 6维状态: [cx, vx, cy, vy, phase0, radius]
  // 转换为11维: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3],
    observed_z_, 0,  // z 用观测值，vz = 0
    ekf_.x[4], omega_est_, ekf_.x[5], 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  // 检查半径是否发散
  double r = ekf_.x[5];
  if (r < 0.15 || r > 0.4) return true;

  // 检查 omega 是否发散
  double omega = std::abs(omega_est_);
  if (omega > 5.0) return true;

  return false;
}

bool OutpostTarget::convergened() const
{
  if (!ekf_initialized_) return false;
  return update_count_ > 10;
}

}  // namespace auto_aim
