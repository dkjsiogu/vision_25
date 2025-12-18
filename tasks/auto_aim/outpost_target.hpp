#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
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
 * 前哨站目标追踪器
 *
 * 前哨站模型：
 * - 三个装甲板，俯视图120度分布
 * - 各层高度不同，但相位和高度的对应关系未知
 * - 同一个旋转中心、角速度、半径
 *
 * 状态向量 (7维):
 * [center_x, vx, center_y, vy, phase0, omega, radius]
 *
 * 关键设计：
 * 1. 用 phase zone 划分装甲板（不匹配，直接按相位区间）
 * 2. z 不放入 EKF，每个 zone 独立记录 z
 * 3. 简单可靠，避免匹配错误
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

  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }
  bool diverged() const;
  bool convergened() const;

  int current_zone() const { return current_zone_; }

  // 获取各 zone 的 z 值列表 (用于传递给Target)
  std::vector<double> zone_z_list() const
  {
    return std::vector<double>(zone_z_, zone_z_ + 3);
  }

  // 获取已初始化的 zone 列表
  std::vector<int> initialized_zones() const
  {
    std::vector<int> zones;
    for (int i = 0; i < 3; i++) {
      if (zone_z_initialized_[i]) {
        zones.push_back(i);
      }
    }
    return zones;
  }

private:
  OutpostState state_ = OutpostState::LOST;

  // EKF: [cx, vx, cy, vy, phase0, omega, radius] (7维，去掉z)
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;
  int update_count_ = 0;

  // 三个 zone 的 z 值（直接记录，不是偏移量）
  double zone_z_[3] = {0, 0, 0};
  bool zone_z_initialized_[3] = {false, false, false};

  int current_zone_ = -1;
  std::chrono::steady_clock::time_point last_update_time_;

  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;
  double outpost_z_gate_ = 0.25;  // m, zone_z 的异常值门限

  // 用于从观测差分辅助初始化/修正 omega（每个 zone 一条时间序列）
  double last_zone_yaw_[3] = {0, 0, 0};
  std::chrono::steady_clock::time_point last_zone_time_[3];
  bool last_zone_yaw_initialized_[3] = {false, false, false};

  void init_ekf(const Armor & armor);

  // 根据预测残差关联 zone (0/1/2)，比固定相位区间更稳
  int match_zone(const Armor & armor) const;

  // 用同一 zone 的观测差分估计 omega（不会替代 EKF，只是给一个更合理的初始/纠偏）
  void update_omega_from_observation(int zone, double armor_yaw, std::chrono::steady_clock::time_point t);

  // 更新指定 zone 的数据
  void update_zone(const Armor & armor, int zone);

  // 计算第 zone 个装甲板的位置
  Eigen::Vector4d armor_xyza(int zone) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
