#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>
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
 * 状态向量 (7维):
 * [center_x, vx, center_y, vy, phase0, omega, radius]
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

  // 获取三个装甲板的 [x, y, z, angle]，z 全部用 observed_z_
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

  // EKF: [cx, vx, cy, vy, phase0, omega, radius] (7维)
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;
  int update_count_ = 0;

  // 观测 z 值（实时更新，滑动平均）
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

  std::chrono::steady_clock::time_point last_update_time_;

  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;

  void init_ekf(const Armor & armor);
  void update_ekf(const Armor & armor);
  void update_pitch_tracking(double pitch);

  // 计算第 i 个装甲板的位置 (i = 0, 1, 2)
  Eigen::Vector4d armor_xyza(int i) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
