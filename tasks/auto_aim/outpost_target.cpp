#include "outpost_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/debug_recorder.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 获取前哨站专用的调试记录器
#define REC tools::DebugRecorder::instance("outpost")

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

  plate_z_ = {0.0, 0.0, 0.0};
  plate_z_valid_ = {false, false, false};
  observed_z_ = 0.0;
  observed_z_valid_ = false;

  pitch_history_.clear();
  pitch_variation_ = 1e10;

  omega_est_ = 0.0;
  unwrapped_phase_history_.clear();
  phase_time_history_.clear();
  unwrapped_phase_accum_ = 0.0;
  window_base_time_valid_ = false;

  phase0_pred_valid_ = false;
  last_pll_time_valid_ = false;

  meas_plate_id_ = 0;
  meas_plate_id_for_update_ = 0;
  meas_valid_ = true;
  last_obs_phase_valid_ = false;
}

void OutpostTarget::align_phase_to_observation_xy(const Armor & armor)
{
  if (!ekf_initialized_) return;

  const double cx = ekf_.x[0], cy = ekf_.x[2];
  const double phase0_pred = ekf_.x[4];
  const double r = outpost_radius_;  // 常量半径

  const double obs_x = armor.xyz_in_world[0];
  const double obs_y = armor.xyz_in_world[1];

  // 观测可能来自三块装甲板中的任意一块：
  // 选取使 (x,y) 残差最小的那一块，记录对应 plate id。
  // 同时记录第二小的残差，用于比值门控。
  int best_i = 0;
  double best_err2 = 1e100;
  double second_err2 = 1e100;
  double err2_list[3] = {0, 0, 0};  // 记录三块板各自的残差

  for (int i = 0; i < 3; i++) {
    const double angle = tools::limit_rad(phase0_pred + i * 2.0 * M_PI / 3.0);
    const double pred_x = cx - r * std::cos(angle);
    const double pred_y = cy - r * std::sin(angle);
    const double dx = obs_x - pred_x;
    const double dy = obs_y - pred_y;
    const double err2 = dx * dx + dy * dy;
    err2_list[i] = err2;
    if (err2 < best_err2) {
      second_err2 = best_err2;
      best_err2 = err2;
      best_i = i;
    } else if (err2 < second_err2) {
      second_err2 = err2;
    }
  }

  // 始终记录本帧用于 EKF update 的候选板号（残差最小者）
  meas_plate_id_for_update_ = best_i;

  // [改进] 智能门控策略：
  const double abs_gate_sq = xy_residual_gate_m_ * xy_residual_gate_m_;
  const double ratio = (second_err2 > 1e-9) ? (best_err2 / second_err2) : 0.0;

  // 1. 基础要求：Best 残差必须小于绝对门限
  bool basic_pass = (best_err2 <= abs_gate_sq);

  // 2. 严格要求：
  //    情况A: 如果 Best 残差非常小（< 绝对门限的 1/4），说明匹配极好，
  //           直接通过，不看比值（解决远距离或视角不好时 ratio 误杀问题）
  //    情况B: 否则，必须满足比值门控（保证区分度）
  bool rigorous_pass = false;
  const double trust_threshold_sq = abs_gate_sq * 0.25;

  if (best_err2 < trust_threshold_sq) {
    rigorous_pass = true;  // 足够可信，豁免 Ratio 检查
  } else {
    rigorous_pass = (ratio < xy_residual_ratio_gate_);  // 否则必须通过 Ratio 检查
  }

  meas_valid_ = basic_pass && rigorous_pass;

  // 只有门控通过时才“对齐/切板”，避免 rejection 帧把随机 best_i 传播到下游，造成瞄点/可视化跳变。
  if (meas_valid_) {
    meas_plate_id_ = best_i;
  }

  // [日志] 记录门控信息
  REC.set("plate_id", best_i);
  REC.set("best_err", std::sqrt(best_err2));
  REC.set("second_err", std::sqrt(second_err2));
  REC.set("err0", std::sqrt(err2_list[0]));
  REC.set("err1", std::sqrt(err2_list[1]));
  REC.set("err2", std::sqrt(err2_list[2]));
  REC.set("ratio", ratio);
  REC.set("basic_pass", basic_pass);
  REC.set("rigor_pass", rigorous_pass);
  REC.set("meas_valid", meas_valid_);

  // 计算最优装甲板的预测位置
  const double best_angle = tools::limit_rad(phase0_pred + best_i * 2.0 * M_PI / 3.0);
  const double pred_x = cx - r * std::cos(best_angle);
  const double pred_y = cy - r * std::sin(best_angle);
  REC.set("pred_x", pred_x);
  REC.set("pred_y", pred_y);
  REC.set("res_x", obs_x - pred_x);
  REC.set("res_y", obs_y - pred_y);

  // [诊断日志] 门控决策细节
  if (!meas_valid_) {
    tools::logger()->debug(
      "[Outpost] REJECT: best_err={:.3f} gate={:.3f} ratio={:.2f} basic={} rigor={}",
      std::sqrt(best_err2), xy_residual_gate_m_, ratio,
      basic_pass ? 1 : 0, rigorous_pass ? 1 : 0);
  }

  // 对前哨站而言，我们始终能给出3块装甲板的预测位置，因此在 TRACKING 时允许 Aimer 进行多板选点。
  // last_id 用于记录当前帧“可信观测”匹配到的装甲板编号（debug/可视化/下游选点参考）。
  jumped = true;
  if (meas_valid_) {
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
  // 使用"predict后、update前"的相位作为预测值，避免 update_ekf() 已经吃入观测导致误差被人为变小。
  const double pred_phase0 = phase0_pred_before_update_;
  const double phase_err = tools::limit_rad(obs_phase0 - pred_phase0);

  // [日志] 记录 PLL 相关数据
  REC.set("obs_phase0", obs_phase0);
  REC.set("phase_err", phase_err);

  // dt 无关：把 pll_Kp_ 视为 omega_dot 的增益（1/s^2 量纲），离散实现 omega += Kp * err * dt
  double dt_pll = 0.0;
  if (last_pll_time_valid_) {
    dt_pll = tools::delta_time(t, last_pll_time_);
  }
  last_pll_time_ = t;
  last_pll_time_valid_ = true;
  REC.set("dt_pll", dt_pll);

  if (dt_pll > 0.0 && dt_pll < 0.12) {
    // [改进] 大幅降低 PLL 增益，防止 omega 跳变
    // 启动期稍大一点帮助点火，稳定期用很小的增益
    const double Kp = (update_count_ < 5) ? (pll_Kp_ * 0.5) : (pll_Kp_ * 0.25);
    const double delta_omega = Kp * phase_err * dt_pll;

    // [关键] omega 变化率限幅：防止单帧跳变太大
    // 启动期允许更快变化（5 rad/s²），稳定期严格（2 rad/s²）
    const double max_rate = (update_count_ < 15) ? 5.0 : 2.0;
    const double max_delta = max_rate * dt_pll;
    omega_est_ += std::clamp(delta_omega, -max_delta, max_delta);
  }

  // ========== 滑窗回归部分：用展开相位序列拟合 omega ==========
  // [改进] 使用预测先验来计算 delta，支持高转速低帧率（抗混叠）
  double dt_step = 0.0;
  if (last_obs_phase_valid_) {
    dt_step = tools::delta_time(t, last_obs_time_);
  }

  // 预测增量：基于当前 omega 估计
  const double pred_dphase = omega_est_ * dt_step;
  // 去中心化的相位差（观测残差）
  const double dphase_centered = tools::limit_rad(obs_phase0 - last_obs_phase_ - pred_dphase);

  // [改进] 动态门限：更宽容，减少误触发导致的回归窗口清空
  // 1.2 rad ≈ 70度，只有非常大的跳变才触发
  const double jump_gate = (std::abs(omega_est_) < 0.5) ? 2.0 : 1.2;
  const bool jump_triggered = last_obs_phase_valid_ && std::abs(dphase_centered) >= jump_gate;

  // [日志] 记录滑窗回归相关数据
  REC.set("dphase_centered", dphase_centered);
  REC.set("jump_gate", jump_gate);
  REC.set("jump_triggered", jump_triggered ? 1 : 0);

  // 如果 centred residual 过大，说明发生了切板误匹配/观测跳变。
  // 这时不应该把"未更新的相位累积值"继续塞进回归窗口，否则回归会被大量水平点拉向 0。
  if (jump_triggered) {
    unwrapped_phase_history_.clear();
    phase_time_history_.clear();
    unwrapped_phase_accum_ = 0.0;
    window_base_time_ = t;
    window_base_time_valid_ = true;

    last_obs_phase_ = obs_phase0;
    last_obs_time_ = t;
    last_obs_phase_valid_ = true;

    unwrapped_phase_history_.push_back(unwrapped_phase_accum_);
    phase_time_history_.push_back(0.0);
    return;
  }

  if (last_obs_phase_valid_) {
    // 核心：累积值 = 预测增量 + 观测残差
    unwrapped_phase_accum_ += (pred_dphase + dphase_centered);
  }
  last_obs_phase_ = obs_phase0;
  last_obs_time_ = t;
  last_obs_phase_valid_ = true;

  // [改进] 记录到历史队列：使用相对时间避免大数精度损失
  // 以窗口第一帧为 T=0，后续帧存储相对时间
  if (unwrapped_phase_history_.empty()) {
    window_base_time_ = t;
    window_base_time_valid_ = true;
  }

  double t_rel = 0.0;
  if (window_base_time_valid_) {
    t_rel = tools::delta_time(t, window_base_time_);
  }

  unwrapped_phase_history_.push_back(unwrapped_phase_accum_);
  phase_time_history_.push_back(t_rel);  // 存相对时间！

  while (unwrapped_phase_history_.size() > OMEGA_WINDOW_SIZE) {
    unwrapped_phase_history_.pop_front();
    phase_time_history_.pop_front();
  }

  // 滑窗回归：拟合 omega = d(phase)/dt
  // 因为时间是相对值（0.0, 0.01, 0.02...），数值精度极高
  REC.set("window_size", static_cast<int>(unwrapped_phase_history_.size()));
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
      // [改进] 更快切换到回归主导，回归更稳定
      const double regress_weight = std::min(1.0, update_count_ / 8.0);

      // 融合时也限制变化率
      const double omega_fused = (1.0 - regress_weight) * omega_est_ + regress_weight * omega_regress;
      const double omega_diff = omega_fused - omega_est_;
      const double max_diff = 0.1;  // 单次融合最大变化 0.1 rad/s
      omega_est_ += std::clamp(omega_diff, -max_diff, max_diff);

      REC.set("omega_regress", omega_regress);
      REC.set("regress_weight", regress_weight);
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

  // 状态: [cx, vx, cy, vy, phase0] (5维)
  // radius 为常量，不放入 EKF
  Eigen::VectorXd x0(5);
  x0 << cx, 0, cy, 0, armor_yaw;

  // ============ P0 设计哲学 ============
  // 前哨站中心静止，速度被完全压死：
  // - 位置：初始 PnP 有误差，给一定不确定性让 EKF 修正
  // - 速度：被强制归零，设极小不确定性（不让 EKF 估计速度）
  // - 相位：第一帧 yaw 可能有误差，给一定不确定性
  Eigen::VectorXd P0_dig(5);
  P0_dig << 0.04, 1e-6, 0.04, 1e-6, 0.4;  // 位置σ=0.2m，速度σ≈0，相位σ=0.63rad
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[4] = tools::limit_rad(c[4]);  // phase0
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  // 初始化观测 z
  plate_z_ = {armor_z, armor_z, armor_z};
  plate_z_valid_ = {true, true, true};
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
  window_base_time_valid_ = false;

  phase0_pred_valid_ = false;
  last_pll_time_valid_ = false;

  tools::logger()->info(
    "[OutpostTarget] Init: cx={:.3f}, cy={:.3f}, phase0={:.3f}, z={:.3f}",
    cx, cy, armor_yaw, armor_z);
}

void OutpostTarget::update_ekf(const Armor & armor)
{
  // 门控失败时：不完全跳过，而是用超大噪声做"软锚定"
  // 这样中心不会飞走，但也不会被错误观测带偏
  const double rejection_sigma = 0.5;  // 门控失败时的大噪声 (m)

  // 观测量：直接用世界系装甲板 (x,y)
  // 这样 phase/omega 由位置序列驱动，不依赖 PnP 的装甲板朝向 yaw（该量在前哨站场景下很可能不稳定/近似常量）。
  Eigen::VectorXd z_obs(2);
  z_obs << armor.xyz_in_world[0], armor.xyz_in_world[1];

  // 状态: [cx, vx, cy, vy, phase0] (5维)
  // 当前帧观测对应的装甲板角度：phase0 + plate_id*120deg
  const double phase0 = ekf_.x[4];
  const double r = outpost_radius_;  // 常量半径
  const int plate_id_for_update = meas_plate_id_for_update_;
  const double angle = tools::limit_rad(phase0 + plate_id_for_update * 2.0 * M_PI / 3.0);

  // 观测雅可比 (2x5)
  // ax = cx - r cos(angle)
  // ay = cy - r sin(angle)
  // d(ax)/d(cx) = 1, d(ax)/d(phase0) = r*sin(angle)
  // d(ay)/d(cy) = 1, d(ay)/d(phase0) = -r*cos(angle)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 5);
  H(0, 0) = 1.0;
  H(0, 4) = r * std::sin(angle);
  H(1, 2) = 1.0;
  H(1, 4) = -r * std::cos(angle);

  // 观测噪声：用距离粗略缩放。pitch 不稳定时整体加大 (x,y) 噪声，避免高度抖动导致的解算漂移拖拽中心/相位。
  double sigma_xy;
  if (!meas_valid_) {
    // 门控失败：用超大噪声做软锚定，防止中心飞走
    sigma_xy = rejection_sigma;
  } else {
    const double dist = std::max(0.0, armor.ypd_in_world[2]);
    sigma_xy = sigma_xy_base_ + sigma_xy_k_ * dist;
    sigma_xy = std::clamp(sigma_xy, sigma_xy_min_, sigma_xy_max_);
    if (!pitch_stable()) {
      sigma_xy *= std::sqrt(pitch_unstable_r_scale_);
    }
  }
  Eigen::VectorXd R_dig(2);
  R_dig << sigma_xy * sigma_xy, sigma_xy * sigma_xy;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    return a - b;
  };

  auto h_func = [&](const Eigen::VectorXd & x) -> Eigen::Vector2d {
    const double cx_ = x[0], cy_ = x[2];
    const double phase0_ = x[4];
    const double angle_ = tools::limit_rad(phase0_ + plate_id_for_update * 2.0 * M_PI / 3.0);
    const double ax = cx_ - r * std::cos(angle_);
    const double ay = cy_ - r * std::sin(angle_);
    return {ax, ay};
  };

  ekf_.update(z_obs, H, R, h_func, z_subtract);

  // ============ Update 后速度约束 ============
  // 前哨站中心静止，EKF update 可能注入速度，必须立即压死
  ekf_.x[1] = 0.0;
  ekf_.x[3] = 0.0;

  // [日志] 记录 EKF 状态
  REC.set("sigma_xy", sigma_xy);
  REC.set("cx", ekf_.x[0]);
  REC.set("vx", ekf_.x[1]);
  REC.set("cy", ekf_.x[2]);
  REC.set("vy", ekf_.x[3]);
  REC.set("phase0", ekf_.x[4]);
  REC.set("radius", outpost_radius_);  // 常量
  REC.set("pitch_var", pitch_variation_);
  REC.set("pitch_stable", pitch_stable());
  REC.set("obs_z", observed_z_);
  REC.set("update_cnt", update_count_);

  // [诊断日志] 每帧打印关键状态，便于定位"中心飞走"原因
  tools::logger()->debug(
    "[Outpost] valid={} σ={:.3f} cx={:.2f} cy={:.2f} vx={:.3f} vy={:.3f} r={:.3f} ω={:.2f}",
    meas_valid_ ? 1 : 0, sigma_xy,
    ekf_.x[0], ekf_.x[2], ekf_.x[1], ekf_.x[3], outpost_radius_, omega_est_);

  // 只有门控通过的帧才计入有效更新次数
  if (meas_valid_) {
    update_count_++;
  }
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

  // [日志] 记录观测原始数据
  REC.set("state", state_ == OutpostState::TRACKING ? 1 : 0);
  REC.set("obs_x", armor.xyz_in_world[0]);
  REC.set("obs_y", armor.xyz_in_world[1]);
  // 注意：obs_z 在 update_ekf() 中会被写成滤波后的 observed_z_。
  // 为避免覆盖导致原始观测丢失，这里额外保留一份 raw。
  REC.set("obs_z_raw", armor.xyz_in_world[2]);
  REC.set("obs_z", armor.xyz_in_world[2]);
  REC.set("obs_yaw", armor.ypr_in_world[0]);
  REC.set("obs_pitch", armor.ypd_in_world[1]);
  REC.set("obs_dist", armor.ypd_in_world[2]);

  // 更新 pitch 追踪（用于判断是否处于高度切换/抖动期）
  update_pitch_tracking(armor.ypd_in_world[1]);

  // 注意：单目 PnP 的 z 对“上中下层”存在歧义。
  // 这里不再用一个 observed_z_ 对所有观测做滑动平均（会把三层混在一起）。
  // 而是在对齐/选板之后，仅在 meas_valid_==true 时更新对应 plate_id 的 z。
  const double obs_z = armor.xyz_in_world[2];

  if (state_ == OutpostState::LOST) {
    init_ekf(armor);
    state_ = OutpostState::TRACKING;
    last_update_time_ = t;

    // 初始化帧也提交调试记录，避免 recorder 时间轴错位
    REC.set("state", 1);  // 刚进入 TRACKING
    REC.set("dt", 0.0);   // 初始化帧没有 dt
    REC.set("omega", omega_est_);
    REC.set("meas_valid", true);
    REC.set("meas_plate_id", meas_plate_id_);
    REC.set("z0", plate_z_[0]);
    REC.set("z1", plate_z_[1]);
    REC.set("z2", plate_z_[2]);
    REC.set("z0_valid", plate_z_valid_[0] ? 1 : 0);
    REC.set("z1_valid", plate_z_valid_[1] ? 1 : 0);
    REC.set("z2_valid", plate_z_valid_[2] ? 1 : 0);
    REC.commit();
    return true;
  }

  // 先 predict 到当前时刻
  double dt = tools::delta_time(t, last_update_time_);
  REC.set("dt", dt);
  if (dt > 0 && dt < 0.1) {
    predict(dt);
  }
  last_update_time_ = t;

  // 观测可能来自不同装甲板：更新前基于 (x,y) 残差对齐 phase0，保持残差连续
  align_phase_to_observation_xy(armor);

  // 更新 z：只更新“可信观测”的那一块板，避免三层互相污染。
  if (meas_valid_) {
    const int zid = meas_plate_id_;
    const double alpha = pitch_stable() ? observed_z_alpha_stable_ : observed_z_alpha_unstable_;
    if (!plate_z_valid_[zid]) {
      plate_z_[zid] = obs_z;
      plate_z_valid_[zid] = true;
    } else {
      plate_z_[zid] = plate_z_[zid] * (1.0 - alpha) + obs_z * alpha;
    }
    observed_z_ = plate_z_[zid];
    observed_z_valid_ = true;
  }

  // 缓存本帧 update 前的预测相位，供 PLL 使用
  phase0_pred_before_update_ = ekf_.x[4];
  phase0_pred_valid_ = true;

  // 更新 EKF
  update_ekf(armor);

  // 用几何相位差分辅助 omega 估计（带跳变门控）
  update_omega_from_observation_xy(armor, t);

  // [日志] 所有数据收集完毕，提交本帧
  REC.set("omega", omega_est_);
  REC.set("meas_plate_id", meas_plate_id_);
  REC.set("meas_plate_id_for_update", meas_plate_id_for_update_);
  REC.set("z0", plate_z_[0]);
  REC.set("z1", plate_z_[1]);
  REC.set("z2", plate_z_[2]);
  REC.set("z0_valid", plate_z_valid_[0] ? 1 : 0);
  REC.set("z1_valid", plate_z_valid_[1] ? 1 : 0);
  REC.set("z2_valid", plate_z_valid_[2] ? 1 : 0);
  REC.commit();

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

  // 状态转移矩阵: [cx, vx, cy, vy, phase0] (5维)
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0},
    {0,  1,  0,  0,  0},
    {0,  0,  1, dt,  0},
    {0,  0,  0,  1,  0},
    {0,  0,  0,  0,  1}
  };
  // clang-format on

  // ============ Q 设计哲学 ============
  // 前哨站中心静止：
  // - v1（位置加速度方差）：极小，中心几乎不动
  // - vphi（相位噪声）：允许 omega 估计有些误差
  double v1 = 0.01;      // 加速度 σ ≈ 0.1 m/s²（原 10，太大导致中心漂移）
  double vphi = 0.2;     // 相位随机游走强度

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,        0},
    {b * v1, c * v1,      0,      0,        0},
    {     0,      0, a * v1, b * v1,        0},
    {     0,      0, b * v1, c * v1,        0},
    {     0,      0,      0,      0, c * vphi}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;

    // ============ 速度约束哲学 ============
    // 前哨站中心是**静止的**，速度估计只会引入漂移：
    // - 极端阻尼：每帧直接归零，彻底压死速度估计
    // - 这样中心位置完全由观测驱动，不会累积漂移
    x_prior[1] = 0.0;  // 直接归零，前哨站中心不会移动
    x_prior[3] = 0.0;

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
  double r = outpost_radius_;  // 常量半径

  // 第 i 个装甲板的角度
  double angle = tools::limit_rad(phase0 + i * 2 * M_PI / 3);

  double armor_x = cx - r * std::cos(angle);
  double armor_y = cy - r * std::sin(angle);
  double armor_z = observed_z_;
  if (i >= 0 && i < 3 && plate_z_valid_[i]) {
    armor_z = plate_z_[i];
  }

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

  // 5维状态: [cx, vx, cy, vy, phase0]
  // 转换为11维: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3],
    observed_z_, 0,  // z 用观测值，vz = 0
    ekf_.x[4], omega_est_, outpost_radius_, 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  // radius 现在是常量，不再需要检查

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
