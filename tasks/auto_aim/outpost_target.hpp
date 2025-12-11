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
  SCANNING,  // 扫描建模中 (识别三层高度)
  TRACKING   // 正常追踪
};

// 高度观测记录
struct HeightObservation
{
  double z;
  std::chrono::steady_clock::time_point t;
};

/**
 * 单层前哨站EKF
 *
 * 状态向量 (9维):
 * [center_x, vx, center_y, vy, z, vz, phase, omega, radius]
 *
 * 每层独立追踪一个装甲板的圆周运动
 */
class OutpostLayerEKF
{
public:
  OutpostLayerEKF() = default;

  void init(const Armor & armor, double radius);
  void predict(double dt);
  void update(const Armor & armor);

  bool is_initialized() const { return initialized_; }
  int update_count() const { return update_count_; }

  // 获取装甲板位置 (x, y, z, phase)
  Eigen::Vector4d armor_xyza() const;

  // 获取共享参数
  double center_x() const { return initialized_ ? ekf_.x[0] : 0; }
  double center_y() const { return initialized_ ? ekf_.x[2] : 0; }
  double omega() const { return initialized_ ? ekf_.x[7] : 0; }

  // 软约束：将共享参数向目标值靠拢
  void apply_shared_constraint(double target_cx, double target_cy, double target_omega, double alpha);

  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }
  bool diverged() const;

private:
  tools::ExtendedKalmanFilter ekf_;
  bool initialized_ = false;
  int update_count_ = 0;
  double radius_ = 0.2765;
};

/**
 * 前哨站目标追踪器
 *
 * 三层独立EKF，共享约束：
 * - 三层的旋转中心(cx, cy)应该一致
 * - 三层的角速度omega应该一致
 * - 每层独立维护自己的phase和z
 *
 * 持续预测：
 * - 即使某层没有观测，也持续predict
 * - 任意时刻都能输出三个装甲板的预测位置
 */
class OutpostTarget
{
public:
  ArmorName name = ArmorName::outpost;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;  // 最近更新的层ID

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

  // 获取三个装甲板的位置列表 (x, y, z, phase)
  // 始终返回3个装甲板，按层0/1/2顺序
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // EKF相关 (兼容Target接口，返回最近更新的层)
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  bool diverged() const;
  bool convergened() const;

  // 调试信息
  int current_layer() const { return current_layer_; }
  double layer_z(int i) const { return (i >= 0 && i < 3) ? layer_z_[i] : 0; }
  bool layer_valid(int i) const { return (i >= 0 && i < 3) ? layer_valid_[i] : false; }
  const OutpostLayerEKF & layer_ekf(int i) const { return layer_ekf_[i]; }

private:
  OutpostState state_ = OutpostState::LOST;

  // 三层独立EKF
  OutpostLayerEKF layer_ekf_[3];

  // 高度层识别 (扫描阶段使用)
  double z_ref_ = 0;
  double layer_z_[3] = {0};  // 三层相对高度
  bool layer_valid_[3] = {false};
  std::deque<HeightObservation> height_observations_;

  // 当前状态
  int current_layer_ = -1;
  std::chrono::steady_clock::time_point last_update_time_;
  std::chrono::steady_clock::time_point scan_start_time_;

  // 配置参数
  double scan_timeout_ = 3.0;
  double cluster_threshold_ = 0.03;
  double layer_gap_min_ = 0.05;
  double layer_gap_max_ = 0.20;
  int min_observations_ = 5;
  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;
  double constraint_alpha_ = 0.3;  // 共享约束强度

  // 扫描建模
  void add_height_observation(double z, std::chrono::steady_clock::time_point t);
  bool try_cluster_heights();

  // 层识别
  int identify_layer(double z) const;

  // 应用共享约束
  void apply_shared_constraints();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
