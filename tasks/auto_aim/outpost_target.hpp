#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <deque>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

enum class OutpostState
{
  LOST,
  TRACKING
};

/**
 * 前哨站目标追踪器 (简化版)
 *
 * 核心思想：
 * - 将三层装甲板的运动压缩成旋转的一层
 * - EKF 只追踪 xy 平面的旋转，忽略 z
 * - z (pitch) 用观测值实时追踪
 * - pitch 变化幅度小于阈值时才允许开火
 *
 * 状态向量:
 * - EKF (5维): [center_x, vx, center_y, vy, phase0]
 * - radius 为常量（不放入 EKF，避免被观测噪声拉偏）
 * - omega 独立估计（不放入 EKF），用于预测 phase0
 */
class OutpostTarget
{
public:
  ArmorName name = ArmorName::outpost;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;

  OutpostTarget() = default;
  explicit OutpostTarget(const std::string & config_path);

  OutpostState state() const { return state_; }
  std::string state_string() const;
  bool is_tracking() const { return state_ == OutpostState::TRACKING; }

  void reset();
  bool update(const Armor & armor, std::chrono::steady_clock::time_point t);
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  // 获取三个装甲板的 [x, y, z, angle]
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 转换为 11 维状态（兼容 Target 接口）
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }

  bool diverged() const;
  bool convergened() const;

  // 获取当前观测 z
  double observed_z() const { return observed_z_; }

  // pitch 稳定性：最近 pitch 变化幅度是否小于阈值
  bool pitch_stable() const;
  double pitch_variation() const { return pitch_variation_; }

private:
  OutpostState state_ = OutpostState::LOST;

  // EKF: [cx, vx, cy, vy, phase0] (5维)
  // radius 为常量，不放入 EKF（避免被观测噪声拉偏）
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;
  int update_count_ = 0;

  // omega 独立估计（不放入 EKF），用于预测 phase0
  // 使用 PLL + 滑窗回归：比差分+EMA更稳定
  double omega_est_ = 0.0;
  double omega_max_abs_ = 2.51;

  // PLL 参数：omega += Kp * phase_error
  double pll_Kp_ = 8.0;  // 比例增益，越大收敛越快但越容易振荡

  // PLL 辅助状态：缓存本帧 update 前的预测相位，以及用于 dt 计算的时间戳
  double phase0_pred_before_update_ = 0.0;
  bool phase0_pred_valid_ = false;
  std::chrono::steady_clock::time_point last_pll_time_;
  bool last_pll_time_valid_ = false;

  // 滑窗回归：维护展开相位序列，用最小二乘拟合 omega
  std::deque<double> unwrapped_phase_history_;
  std::deque<double> phase_time_history_;  // 存相对时间（避免大数精度损失）
  double unwrapped_phase_accum_ = 0.0;     // 展开相位累积值
  std::chrono::steady_clock::time_point window_base_time_;  // 滑窗基准时间
  bool window_base_time_valid_ = false;
  static constexpr size_t OMEGA_WINDOW_SIZE = 10;

  // z 估计：单目无法可靠判定“上中下层”，但在 TRACKING 状态下我们能稳定地给出
  // 三块装甲板的角度 id（0/1/2）。因此对每个 id 维护一条 z 估计，并且只在
  // meas_valid_==true 时更新对应 id，避免三层 z 互相污染。
  std::array<double, 3> plate_z_{{0.0, 0.0, 0.0}};
  std::array<bool, 3> plate_z_valid_{{false, false, false}};

  // 当前用于下游（瞄准/弹道）的一块装甲板 z（通常等于 plate_z_[meas_plate_id_]）
  double observed_z_ = 0.0;
  bool observed_z_valid_ = false;

  // pitch 追踪（用于判断稳定性）
  std::deque<double> pitch_history_;
  static constexpr size_t PITCH_HISTORY_SIZE = 10;
  double pitch_variation_ = 1e10;  // 最近 pitch 变化幅度
  double pitch_stable_threshold_ = 0.05;  // rad, 约 3 度

  // pitch 不稳定时的处理参数
  // - EKF：增大 pitch 观测噪声，避免高度切换/误差抖动拖拽平面状态
  // - z：使用更保守的滑动平均系数，避免多高度混入导致 observed_z_ 振荡
  double pitch_unstable_r_scale_ = 25.0;
  double observed_z_alpha_stable_ = 0.7;
  double observed_z_alpha_unstable_ = 0.2;

  // (x,y) 观测噪声参数：sigma_xy = clamp(base + k * dist, min, max)
  // [修复] 增大噪声，更现实地反映 PnP 误差（原值太乐观导致过度信任观测）
  double sigma_xy_base_ = 0.05;   // 5cm 基础噪声（原 1.5cm）
  double sigma_xy_k_ = 0.01;      // 每米增加 1cm（原 0.15cm）
  double sigma_xy_min_ = 0.03;    // 最小 3cm（原 0.6cm）
  double sigma_xy_max_ = 0.15;    // 最大 15cm（原 6cm）

  // (x,y) 残差门控：
  // - 绝对门控：最小残差若仍然过大，则本帧跳过 EKF 更新/omega 更新
  // - 比值门控：最小残差必须显著优于第二名，避免中心漂移时三个都大但仍选出一个
  // [修复] 放宽门控：前哨站半径0.28m，PnP误差+z层差异，0.30m更合理
  double xy_residual_gate_m_ = 0.30;          // 30cm（原 25cm，考虑z层差异带来的透视误差）
  double xy_residual_ratio_gate_ = 0.8;       // 比值门控（区分度）

  std::chrono::steady_clock::time_point last_update_time_;

  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;

  void init_ekf(const Armor & armor);
  void update_ekf(const Armor & armor);
  void update_pitch_tracking(double pitch);

  // 观测辅助：
  // - 基于 (x,y) 残差选择/对齐当前观测对应的装甲板（避免 120° 切板跳变导致相位不连续）
  // - 用中心->装甲板向量推回观测相位，辅助估计 omega
  int meas_plate_id_ = 0;
  // 本帧用于 EKF update 的候选装甲板编号（始终取残差最小者）。
  // 注意：当 meas_valid_==false 时，我们不会把该编号传播给下游（last_id/meas_plate_id_），
  // 但 EKF 的观测模型仍使用该候选编号以尽可能匹配当前观测。
  int meas_plate_id_for_update_ = 0;
  bool meas_valid_ = true;
  double last_obs_phase_ = 0.0;
  std::chrono::steady_clock::time_point last_obs_time_;
  bool last_obs_phase_valid_ = false;

  void align_phase_to_observation_xy(const Armor & armor);
  void update_omega_from_observation_xy(const Armor & armor, std::chrono::steady_clock::time_point t);

  // 计算第 i 个装甲板的位置 (i = 0, 1, 2)
  Eigen::Vector4d armor_xyza(int i) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
