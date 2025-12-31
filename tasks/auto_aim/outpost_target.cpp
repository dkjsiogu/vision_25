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
  if (yaml["outpost_omega_est_max_abs"]) {
    omega_est_max_abs_ = yaml["outpost_omega_est_max_abs"].as<double>();
  } else if (yaml["outpost_omega_max_abs"]) {
    // 兼容旧配置：单一限幅同时设置两个
    omega_est_max_abs_ = yaml["outpost_omega_max_abs"].as<double>();
    omega_regress_max_abs_ = omega_est_max_abs_;
  }
  if (yaml["outpost_omega_regress_max_abs"]) {
    omega_regress_max_abs_ = yaml["outpost_omega_regress_max_abs"].as<double>();
  }
  if (yaml["outpost_pll_Kp"]) {
    pll_Kp_ = yaml["outpost_pll_Kp"].as<double>();
  }
  if (yaml["outpost_omega_prior_abs"]) {
    omega_prior_abs_ = yaml["outpost_omega_prior_abs"].as<double>();
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
  omega_regress_ema_ = 0.0;
  omega_regress_ema_valid_ = false;
  omega_direction_ = 0;
  omega_sign_count_ = 0;

  // omega 学习状态机重置
  omega_learn_state_ = OmegaLearnState::ACQUIRE;
  omega_locked_value_ = 0.0;
  omega_lock_frame_count_ = 0;
  omega_unlock_frame_count_ = 0;
  last_sigma_regress_ = 999.0;
  last_jump_triggered_ = false;

  // z 三层模型重置
  layer_z_ = {0.0, 0.0, 0.0};
  plate_to_layer_ = {-1, -1, -1};
  for (auto & row : plate_layer_votes_) row = {0, 0, 0};
  layer_initialized_ = false;
  layer_total_votes_ = 0;

  phase0_pred_valid_ = false;
  last_pll_time_valid_ = false;

  meas_plate_id_ = 0;
  meas_plate_id_for_update_ = 0;
  meas_valid_ = true;
  plate_switch_candidate_ = -1;
  plate_switch_count_ = 0;
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

  // 门控失败：不累计滞回状态，避免把旧候选带到下一次 valid 帧。
  if (!meas_valid_) {
    plate_switch_candidate_ = -1;
    plate_switch_count_ = 0;
  }

  // 板号切换滞回保护：防止误匹配导致的频繁切板
  // 只有连续 N 帧候选板一致且与当前板不同时，才允许切板
  if (meas_valid_) {
    if (best_i == meas_plate_id_) {
      // 仍是当前板，重置候选
      plate_switch_candidate_ = -1;
      plate_switch_count_ = 0;
    } else if (best_i == plate_switch_candidate_) {
      // 候选板连续确认
      plate_switch_count_++;
      if (plate_switch_count_ >= PLATE_SWITCH_HYSTERESIS) {
        meas_plate_id_ = best_i;
        plate_switch_candidate_ = -1;
        plate_switch_count_ = 0;
      }
    } else {
      // 新候选板
      plate_switch_candidate_ = best_i;
      plate_switch_count_ = 1;
    }
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
    // 对外暴露稳定板号，避免短时误匹配/切板抖动影响下游。
    last_id = meas_plate_id_;
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

  // 内部相位/omega 估计使用“本帧几何残差最小”的候选板号，避免滞回造成的 120° 映射错误。
  const int plate_id_for_phase = meas_plate_id_for_update_;

  // 用中心->装甲板的几何关系反推出"观测相位"（不依赖 PnP 的装甲板朝向 yaw）。
  // armor_x = cx - r cos(phase) => (cx - armor_x, cy - armor_y) 与 (cos,sin) 同向
  const double obs_phase_plate = std::atan2(cy - obs_y, cx - obs_x);
  // 转换为"plate0"的相位观测，避免切板造成的 120° 跳变污染 omega
  const double obs_phase0 =
    tools::limit_rad(obs_phase_plate - plate_id_for_phase * 2.0 * M_PI / 3.0);

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

  // dt 异常：直接重置回归窗口，避免 dt 缩放导致一次性大跳。
  if (last_obs_phase_valid_ && (dt_step <= 0.0 || dt_step > 0.12)) {
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
    omega_regress_ema_ = 0.0;
    omega_regress_ema_valid_ = false;
    return;
  }

  // 预测增量：基于当前 omega 估计
  const double pred_dphase = omega_est_ * dt_step;
  // 去中心化的相位差（观测残差）
  const double dphase_centered = tools::limit_rad(obs_phase0 - last_obs_phase_ - pred_dphase);

  // [改进] 动态门限：更宽容，减少误触发导致的回归窗口清空
  // 1.2 rad ≈ 70度，只有非常大的跳变才触发
  const double jump_gate = (std::abs(omega_est_) < 0.5) ? 2.0 : 1.2;
  const bool jump_triggered = last_obs_phase_valid_ && std::abs(dphase_centered) >= jump_gate;
  last_jump_triggered_ = jump_triggered;  // 缓存给状态机用

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
    omega_regress_ema_ = 0.0;
    omega_regress_ema_valid_ = false;
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
      double omega_regress_raw = num / den;

      // [防护] NaN/inf 检测：异常值直接跳过本帧回归更新
      if (!std::isfinite(omega_regress_raw)) {
        REC.set("omega_regress_raw", 0.0);
        REC.set("omega_regress", 0.0);
        REC.set("omega_regress_ema", omega_regress_ema_);
        REC.set("regress_weight", 0.0);
        REC.set("prior_weight", 0.0);
        REC.set("regress_sigma", 0.0);
        REC.set("regress_hit_limit", 1);  // 当作异常帧
        return;
      }

      // [改进] 计算回归不确定度（斜率标准误差）
      // sigma = sqrt(MSE / Sxx)，其中 MSE = SSE / (n-2)，Sxx = sum((t-t_mean)^2) = den
      double sse = 0.0;
      for (size_t i = 0; i < n; i++) {
        double pred = phi_mean + omega_regress_raw * (phase_time_history_[i] - t_mean);
        double err = unwrapped_phase_history_[i] - pred;
        sse += err * err;
      }
      const double mse = (n > 2) ? sse / (n - 2) : sse;
      const double sigma_regress = std::sqrt(mse / den);
      last_sigma_regress_ = sigma_regress;  // 缓存给状态机用

      // [改进] 回归限幅：用 omega_regress_max_abs_ 裁掉极端值
      const bool regress_hit_limit =
        std::abs(omega_regress_raw) >= omega_regress_max_abs_ * 0.95;
      double omega_regress =
        std::clamp(omega_regress_raw, -omega_regress_max_abs_, omega_regress_max_abs_);

      // [修复] 对 omega_regress 做 EMA 平滑，减少单次回归噪声
      const double ema_alpha = 0.3;
      if (!omega_regress_ema_valid_) {
        omega_regress_ema_ = omega_regress;
        omega_regress_ema_valid_ = true;
      } else {
        omega_regress_ema_ = ema_alpha * omega_regress + (1.0 - ema_alpha) * omega_regress_ema_;
      }

      // [改进] 更快切换到回归主导
      const double regress_weight = std::min(1.0, update_count_ / 8.0);

      // [改进] 贝叶斯风格先验权重：基于回归不确定度连续调整
      // sigma_regress 大 → 回归不可靠 → 更信先验
      // sigma_regress 小 → 回归可靠 → 更信观测
      double prior_weight = 0.0;
      if (update_count_ > 10) {
        // 贝叶斯公式：w_prior = σ_obs² / (σ_obs² + σ_prior²)
        // σ_prior 设为 0.3 rad/s（先验的不确定度，即我们对 2.51 的信心）
        const double sigma_prior = 0.3;
        const double sigma_obs_sq = sigma_regress * sigma_regress;
        const double sigma_prior_sq = sigma_prior * sigma_prior;
        prior_weight = sigma_obs_sq / (sigma_obs_sq + sigma_prior_sq);

        // 限幅：最多 30%，避免先验主导
        prior_weight = std::clamp(prior_weight, 0.0, 0.3);

        // 撞限幅时额外加权
        if (regress_hit_limit) {
          prior_weight = std::max(prior_weight, 0.15);
        }
      }

      // [改进] 方向锁定机制：基于 omega_regress_ema 的符号
      // 不再用 omega_est_（因为它受 prior 影响，会自我强化）
      const int current_sign = (omega_regress_ema_ >= 0) ? 1 : -1;

      if (omega_direction_ == 0) {
        // 方向未锁定：累计同符号计数
        if (current_sign == omega_sign_count_ / std::max(std::abs(omega_sign_count_), 1)) {
          // 符号一致，继续累计
        } else if (omega_sign_count_ == 0) {
          // 首次观测
        }
        omega_sign_count_ = (current_sign > 0)
          ? std::max(1, omega_sign_count_ + 1)
          : std::min(-1, omega_sign_count_ - 1);

        if (std::abs(omega_sign_count_) >= DIRECTION_LOCK_THRESHOLD) {
          omega_direction_ = (omega_sign_count_ > 0) ? 1 : -1;
          tools::logger()->info("[Outpost] Direction locked: {}", omega_direction_ > 0 ? "+" : "-");
        }
      } else {
        // 方向已锁定：检测是否需要翻转
        if (current_sign != omega_direction_) {
          omega_sign_count_++;
          if (omega_sign_count_ >= DIRECTION_LOCK_THRESHOLD * 2) {
            // 需要连续更多帧才能翻转（抵抗噪声）
            omega_direction_ = current_sign;
            omega_sign_count_ = 0;
            tools::logger()->info("[Outpost] Direction flipped: {}", omega_direction_ > 0 ? "+" : "-");
          }
        } else {
          omega_sign_count_ = 0;  // 重置翻转计数
        }
      }

      // 先验值：使用锁定方向（如果已锁定），否则用回归方向
      const int prior_sign = (omega_direction_ != 0) ? omega_direction_ : current_sign;
      const double omega_prior_signed = prior_sign * omega_prior_abs_;
      const double omega_with_prior =
        (1.0 - prior_weight) * omega_regress_ema_ + prior_weight * omega_prior_signed;

      // [修复] 使用带先验的 omega 进行融合
      const double omega_fused = (1.0 - regress_weight) * omega_est_ + regress_weight * omega_with_prior;
      const double omega_diff = omega_fused - omega_est_;
      const double omega_fuse_rate = 15.0;  // 提高到 15 rad/s²，加快收敛
      const double max_diff = omega_fuse_rate * dt_step;
      omega_est_ += std::clamp(omega_diff, -max_diff, max_diff);

      REC.set("omega_regress_raw", omega_regress_raw);
      REC.set("omega_regress", omega_regress);
      REC.set("omega_regress_ema", omega_regress_ema_);
      REC.set("regress_weight", regress_weight);
      REC.set("prior_weight", prior_weight);
      REC.set("regress_sigma", sigma_regress);
      REC.set("regress_hit_limit", regress_hit_limit ? 1 : 0);
      REC.set("omega_direction", omega_direction_);
    }
  }

  // [改进] 最终限幅：用 omega_est_max_abs_ 给足动态余量
  omega_est_ = std::clamp(omega_est_, -omega_est_max_abs_, omega_est_max_abs_);
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
  // 采用"强先验 + 慢漂移"模型：
  // - 位置：初始 PnP 有误差，给一定不确定性让 EKF 修正
  // - 速度：从 0 开始，允许缓慢变化（配合 Q 矩阵和 predict 中的阻尼）
  // - 相位：第一帧 yaw 可能有误差，给一定不确定性
  Eigen::VectorXd P0_dig(5);
  P0_dig << 0.04, 0.01, 0.04, 0.01, 0.4;  // 位置σ=0.2m，速度σ=0.1m/s，相位σ=0.63rad
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[4] = tools::limit_rad(c[4]);  // phase0
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  // 初始化观测 z：只初始化第一帧“实际看到的那一块板”。
  // 不能假设第一帧一定是 plate 0，否则会把 z 写错板，后续弹道必然偏。
  int best_i = 0;
  double best_err2 = 1e100;
  for (int i = 0; i < 3; i++) {
    const double angle = tools::limit_rad(armor_yaw + i * 2.0 * M_PI / 3.0);
    const double pred_x = cx - outpost_radius_ * std::cos(angle);
    const double pred_y = cy - outpost_radius_ * std::sin(angle);
    const double dx = armor_x - pred_x;
    const double dy = armor_y - pred_y;
    const double err2 = dx * dx + dy * dy;
    if (err2 < best_err2) {
      best_err2 = err2;
      best_i = i;
    }
  }

  plate_z_ = {0.0, 0.0, 0.0};
  plate_z_valid_ = {false, false, false};
  plate_z_[best_i] = armor_z;
  plate_z_valid_[best_i] = true;
  observed_z_ = armor_z;
  observed_z_valid_ = true;

  // 初始化 pitch 追踪
  double pitch = armor.ypd_in_world[1];
  pitch_history_.clear();
  pitch_history_.push_back(pitch);

  // 初始化 plate-id / 相位差分缓存，帮助 omega 更快点火
  meas_plate_id_ = best_i;
  meas_plate_id_for_update_ = best_i;
  meas_valid_ = true;
  last_obs_phase_ = tools::limit_rad(std::atan2(cy - armor_y, cx - armor_x));
  last_obs_time_ = std::chrono::steady_clock::time_point{};  // 由首次 update() 写入
  last_obs_phase_valid_ = false;  // 下一帧开始做差分

  // 前哨站可始终输出 3 板预测，进入 TRACKING 即视为 jumped。
  jumped = true;
  last_id = best_i;

  // [改进] 初始化 omega 为 0，让观测决定方向
  // 不再用先验初始化，因为方向未知
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
  // 说明：armor.xyz_in_world 是“以云台/相机为原点的目标相对位置向量，在世界方向系下表达”。
  // 该量会随车体平移而变化，因此在生产端不能假设前哨站中心 (cx,cy) 永远不动。
  // 这里采用“强先验 + 慢变化”的模型：允许 EKF 更新中心与速度，但过程噪声保持较小。

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

  // 观测噪声：用距离粗略缩放。
  // [修复] 对于前哨站，pitch 跳变是装甲板切换的正常结果，不应惩罚 (x,y) 观测。
  // pitch_stable 仅用于控制 z 值的更新速率（在 update() 中处理）。
  double sigma_xy;
  if (!meas_valid_) {
    // 门控失败：用大噪声减少对 phase0 的影响
    sigma_xy = 0.5;  // 50cm，门控失败时不信任观测
  } else {
    const double dist = std::max(0.0, armor.ypd_in_world[2]);
    sigma_xy = sigma_xy_base_ + sigma_xy_k_ * dist;
    sigma_xy = std::clamp(sigma_xy, sigma_xy_min_, sigma_xy_max_);
    // 注意：不再对 pitch_unstable 情况惩罚 (x,y) 观测
    // 前哨站的 pitch 跳变是三层高度切换的几何效应，(x,y) 观测仍然有效
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

  // [改进] 门控失败时额外抑制速度，防止误匹配帧累积速度误差
  if (!meas_valid_) {
    ekf_.x[1] *= 0.8;  // vx 衰减 20%
    ekf_.x[3] *= 0.8;  // vy 衰减 20%
  }

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
    // [重要] 必须在第一次 commit() 前设置所有可能的字段，否则 CSV 表头会缺列
    REC.set("state", 1);  // 刚进入 TRACKING
    REC.set("dt", 0.0);   // 初始化帧没有 dt
    REC.set("omega", omega_est_);
    REC.set("meas_valid", true);
    REC.set("meas_plate_id", meas_plate_id_);
    REC.set("meas_plate_id_for_update", meas_plate_id_);
    REC.set("z0", plate_z_[0]);
    REC.set("z1", plate_z_[1]);
    REC.set("z2", plate_z_[2]);
    REC.set("z0_valid", plate_z_valid_[0] ? 1 : 0);
    REC.set("z1_valid", plate_z_valid_[1] ? 1 : 0);
    REC.set("z2_valid", plate_z_valid_[2] ? 1 : 0);
    // align_phase_to_observation_xy 的字段
    REC.set("plate_id", meas_plate_id_);
    REC.set("best_err", 0.0);
    REC.set("second_err", 0.0);
    REC.set("err0", 0.0);
    REC.set("err1", 0.0);
    REC.set("err2", 0.0);
    REC.set("ratio", 0.0);
    REC.set("basic_pass", true);
    REC.set("rigor_pass", true);
    REC.set("pred_x", armor.xyz_in_world[0]);
    REC.set("pred_y", armor.xyz_in_world[1]);
    REC.set("res_x", 0.0);
    REC.set("res_y", 0.0);
    // update_ekf 的字段
    REC.set("sigma_xy", 0.0);
    REC.set("cx", ekf_.x[0]);
    REC.set("vx", ekf_.x[1]);
    REC.set("cy", ekf_.x[2]);
    REC.set("vy", ekf_.x[3]);
    REC.set("phase0", ekf_.x[4]);
    REC.set("radius", outpost_radius_);
    REC.set("pitch_var", pitch_variation_);
    REC.set("pitch_stable", pitch_stable());
    REC.set("obs_z", observed_z_);
    REC.set("update_cnt", update_count_);
    // update_omega_from_observation_xy 的字段
    REC.set("obs_phase0", 0.0);
    REC.set("phase_err", 0.0);
    REC.set("dt_pll", 0.0);
    REC.set("dphase_centered", 0.0);
    REC.set("jump_gate", 0.0);
    REC.set("jump_triggered", 0);
    REC.set("window_size", 0);
    REC.set("omega_regress_raw", 0.0);
    REC.set("omega_regress", 0.0);
    REC.set("regress_weight", 0.0);
    REC.set("prior_weight", 0.0);
    REC.set("regress_sigma", 0.0);
    REC.set("regress_hit_limit", 0);
    REC.set("omega_direction", 0);
    // omega 学习状态机的字段
    REC.set("omega_learn_state", 0);
    REC.set("omega_locked_value", 0.0);
    REC.set("omega_lock_frames", 0);
    // z 三层模型的字段
    REC.set("layer_z0", 0.0);
    REC.set("layer_z1", 0.0);
    REC.set("layer_z2", 0.0);
    REC.set("plate_to_layer0", -1);
    REC.set("plate_to_layer1", -1);
    REC.set("plate_to_layer2", -1);
    REC.set("layer_initialized", 0);
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
    // z 归属应跟随本帧观测所对应的候选板号；切板滞回只影响对外暴露板号。
    const int zid = meas_plate_id_for_update_;
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

  // omega 学习状态机更新
  update_omega_learn_state(last_sigma_regress_, last_jump_triggered_);

  // z 三层模型学习（仅在有效帧）
  if (meas_valid_) {
    update_layer_model(obs_z, meas_plate_id_for_update_);
  }

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
  // 采用“强先验 + 慢变化”的中心运动模型：
  // - v1（位置加速度方差）：偏小，使中心变化更平滑（但不强行锁死）
  // - vphi（相位噪声）：允许 omega 估计有些误差
  double v1 = 0.01;      // 加速度 σ ≈ 0.1 m/s²；后续可结合日志再调
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

    // [改进] 速度阻尼：防止 vx/vy 在没有观测校正时持续累积
    // 阻尼因子 0.95：每帧衰减 5%，约 20 帧（0.3s）后衰减到 1/3
    constexpr double vel_damping = 0.95;
    x_prior[1] *= vel_damping;  // vx
    x_prior[3] *= vel_damping;  // vy

    // [改进] 速度限幅：前哨站中心相对云台的移动速度不应过大
    // 0.3 m/s 足够覆盖正常车体移动（步兵典型速度 2-3 m/s，但相对距离变化更慢）
    constexpr double max_vel = 0.3;
    x_prior[1] = std::clamp(x_prior[1], -max_vel, max_vel);
    x_prior[3] = std::clamp(x_prior[3], -max_vel, max_vel);

    // phase0 由 omega 外推：锁定时用 omega_locked_value_，否则用 omega_est_
    const double omega_for_predict =
      (omega_learn_state_ == OmegaLearnState::LOCKED) ? omega_locked_value_ : omega_est_;
    x_prior[4] = tools::limit_rad(x_prior[4] + omega_for_predict * dt);
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

  // 高度处理：
  // - 只有观测到并初始化过的板才使用其独立 z
  // - 未初始化的板不做“硬猜偏移”，否则会把错误 z 传到下游弹道，导致必然打不中
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

// ============================================================================
// omega 学习状态机：ACQUIRE → LOCKED
// ============================================================================
void OutpostTarget::update_omega_learn_state(double sigma_regress, bool jump_triggered)
{
  // 锁定条件检测
  bool lock_condition =
    meas_valid_ &&
    !jump_triggered &&
    sigma_regress < omega_lock_sigma_thr_ &&
    omega_regress_ema_valid_ &&
    std::abs(omega_est_ - omega_regress_ema_) < omega_lock_delta_thr_;

  // 解锁条件检测
  bool unlock_condition =
    jump_triggered ||
    !meas_valid_ ||
    sigma_regress > omega_lock_sigma_thr_ * 2.0;

  if (omega_learn_state_ == OmegaLearnState::ACQUIRE) {
    // 点火期：检测是否可以锁定
    if (lock_condition) {
      omega_lock_frame_count_++;
      if (omega_lock_frame_count_ >= OMEGA_LOCK_THRESHOLD) {
        // 锁定！
        omega_learn_state_ = OmegaLearnState::LOCKED;
        omega_locked_value_ = omega_est_;
        omega_unlock_frame_count_ = 0;
        tools::logger()->info(
          "[Outpost] omega LOCKED: {:.3f} rad/s (after {} frames)",
          omega_locked_value_, omega_lock_frame_count_);
      }
    } else {
      omega_lock_frame_count_ = 0;
    }
  } else {
    // 锁定期：检测是否需要解锁
    if (unlock_condition) {
      omega_unlock_frame_count_++;
      if (omega_unlock_frame_count_ >= OMEGA_UNLOCK_THRESHOLD) {
        // 解锁！
        omega_learn_state_ = OmegaLearnState::ACQUIRE;
        omega_lock_frame_count_ = 0;
        tools::logger()->info("[Outpost] omega UNLOCKED (after {} bad frames)", omega_unlock_frame_count_);
      }
    } else {
      omega_unlock_frame_count_ = 0;
    }
  }

  // 日志
  REC.set("omega_learn_state", omega_learn_state_ == OmegaLearnState::LOCKED ? 1 : 0);
  REC.set("omega_locked_value", omega_locked_value_);
  REC.set("omega_lock_frames", omega_lock_frame_count_);
}

// ============================================================================
// z 三层模型学习
// ============================================================================
void OutpostTarget::update_layer_model(double z_obs, int plate_id)
{
  if (plate_id < 0 || plate_id >= 3) return;

  // 阶段1：初始化层中心（用前 N 帧的观测）
  if (!layer_initialized_) {
    layer_total_votes_++;

    // 简单策略：直接用观测更新对应 plate 的 layer 中心
    // 初始假设：plate 0 → layer 0, plate 1 → layer 1, plate 2 → layer 2
    double alpha = 0.3;
    if (layer_total_votes_ <= 3) {
      // 前几帧直接赋值
      layer_z_[plate_id] = z_obs;
    } else {
      layer_z_[plate_id] = layer_z_[plate_id] * (1.0 - alpha) + z_obs * alpha;
    }

    // 累计足够帧数后，对层中心排序
    if (layer_total_votes_ >= LAYER_INIT_FRAMES) {
      // 按高度排序：layer 0 = 最高, layer 2 = 最低
      std::array<std::pair<double, int>, 3> sorted;
      for (int i = 0; i < 3; i++) sorted[i] = {layer_z_[i], i};
      std::sort(sorted.begin(), sorted.end(), std::greater<>());

      // 重新映射
      std::array<double, 3> new_layer_z;
      for (int i = 0; i < 3; i++) {
        new_layer_z[i] = sorted[i].first;
      }
      layer_z_ = new_layer_z;

      layer_initialized_ = true;
      tools::logger()->info(
        "[Outpost] Layer initialized: z = [{:.3f}, {:.3f}, {:.3f}]",
        layer_z_[0], layer_z_[1], layer_z_[2]);
    }
  } else {
    // 阶段2：最近邻分配 + 更新层中心 + 投票

    // 找最近的层
    int nearest_layer = 0;
    double min_dist = std::abs(z_obs - layer_z_[0]);
    for (int k = 1; k < 3; k++) {
      double dist = std::abs(z_obs - layer_z_[k]);
      if (dist < min_dist) {
        min_dist = dist;
        nearest_layer = k;
      }
    }

    // 更新层中心（慢速 EMA）
    double alpha = 0.1;
    layer_z_[nearest_layer] = layer_z_[nearest_layer] * (1.0 - alpha) + z_obs * alpha;

    // 给 plate_id 投票
    plate_layer_votes_[plate_id][nearest_layer]++;

    // 检查是否可以锁定映射
    int total_votes_for_plate = 0;
    int max_votes = 0;
    int max_layer = -1;
    for (int k = 0; k < 3; k++) {
      total_votes_for_plate += plate_layer_votes_[plate_id][k];
      if (plate_layer_votes_[plate_id][k] > max_votes) {
        max_votes = plate_layer_votes_[plate_id][k];
        max_layer = k;
      }
    }

    // 锁定条件：票数够多 且 占比够高
    if (total_votes_for_plate >= LAYER_LOCK_VOTES &&
        max_votes >= total_votes_for_plate * LAYER_LOCK_RATIO &&
        plate_to_layer_[plate_id] < 0) {
      plate_to_layer_[plate_id] = max_layer;
      tools::logger()->info(
        "[Outpost] Plate {} -> Layer {} locked (votes: {}/{})",
        plate_id, max_layer, max_votes, total_votes_for_plate);
    }
  }

  // 日志
  REC.set("layer_z0", layer_z_[0]);
  REC.set("layer_z1", layer_z_[1]);
  REC.set("layer_z2", layer_z_[2]);
  REC.set("plate_to_layer0", plate_to_layer_[0]);
  REC.set("plate_to_layer1", plate_to_layer_[1]);
  REC.set("plate_to_layer2", plate_to_layer_[2]);
  REC.set("layer_initialized", layer_initialized_ ? 1 : 0);
}

}  // namespace auto_aim
