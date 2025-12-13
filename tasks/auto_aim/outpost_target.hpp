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

// 前哨站追踪状态
enum class OutpostState
{
  LOST,      // 丢失
  TRACKING   // 追踪中
};

/**
 * 前哨站目标追踪器
 *
 * 核心思想：
 * - 单一EKF维护旋转模型 [cx, vx, cy, vy, phase, omega, radius]
 * - 三个装甲板强制120度分布
 * - 用观测的z值来区分是哪一层
 * - 三层各自维护高度z（相对稳定）
 *
 * 状态向量 (7维):
 * [center_x, vx, center_y, vy, phase, omega, radius]
 *
 * 三个装甲板位置:
 * - 层0: (cx - r*cos(phase),        cy - r*sin(phase),        z[0])
 * - 层1: (cx - r*cos(phase + 2π/3), cy - r*sin(phase + 2π/3), z[1])
 * - 层2: (cx - r*cos(phase + 4π/3), cy - r*sin(phase + 4π/3), z[2])
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

  // 状态查询
  OutpostState state() const { return state_; }
  std::string state_string() const;
  bool is_tracking() const { return state_ == OutpostState::TRACKING; }

  // 核心接口
  void reset();
  bool update(const Armor & armor, std::chrono::steady_clock::time_point t);
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  // 获取三个装甲板的位置列表 (x, y, z, angle)
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // EKF相关 (兼容Target接口)
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }
  bool diverged() const;
  bool convergened() const;

  // 调试信息
  int current_layer() const { return current_layer_; }
  double layer_z(int i) const { return (i >= 0 && i < 3) ? layer_z_[i] : 0; }

private:
  OutpostState state_ = OutpostState::LOST;

  // 单一EKF: [cx, vx, cy, vy, phase, omega, radius]
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;
  int update_count_ = 0;

  // 三层高度 (用PnP的z区分层，虽然不准但稳定)
  double layer_z_[3] = {0, 0, 0};
  bool layer_initialized_[3] = {false, false, false};
  int layer_observation_count_[3] = {0, 0, 0};

  // 每层的相位偏移 (相对于phase0，从观测中学习)
  // 三层应该接近 0°, 120°, 240° 的某种排列
  double layer_phase_offset_[3] = {0, 0, 0};

  // 高度聚类参数
  double z_cluster_threshold_ = 0.05;  // 高度聚类阈值

  // 当前状态
  int current_layer_ = -1;
  std::chrono::steady_clock::time_point last_update_time_;

  // 配置参数
  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;

  // 初始化EKF
  void init_ekf(const Armor & armor, int layer);

  // 根据z值识别层 (返回0/1/2，或-1表示需要新建层)
  int identify_layer(double z);

  // 更新指定层
  void update_layer(const Armor & armor, int layer);

  // 计算第i层装甲板的位置
  Eigen::Vector4d armor_xyza(int layer) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
