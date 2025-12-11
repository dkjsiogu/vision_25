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

// 前哨站扫描观测点
struct OutpostObservation
{
  double x, y;           // 推算的旋转中心xy
  double z;              // 原始z值
  double z_relative;     // 相对高度 (z - z_ref)
  double armor_yaw;      // 装甲板朝向
  std::chrono::steady_clock::time_point t;
};

// 前哨站模型
struct OutpostModel
{
  // 旋转中心
  double center_x = 0;
  double center_y = 0;
  double z_ref = 0;  // 高度参考点

  // 旋转参数
  double radius = 0.2765;  // 旋转半径 (m)
  double omega = 0;        // 角速度 (rad/s)
  double phase_at_t0 = 0;  // t0时刻的相位
  std::chrono::steady_clock::time_point t0;

  // 三层高度差 (相对于z_ref)
  double layer_z[3] = {0, 0, 0};  // 下、中、上三层
  bool layer_valid[3] = {false, false, false};

  // 模型状态
  bool is_ready = false;
  int validation_count = 0;
};

// 前哨站追踪状态
enum class OutpostState
{
  LOST,        // 丢失
  SCANNING,    // 扫描建模中
  VALIDATING,  // 验证模型中
  TRACKING     // 正常追踪
};

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
  bool is_ready() const { return model_.is_ready; }

  // 核心接口
  void reset();
  bool update(const Armor & armor, std::chrono::steady_clock::time_point t);
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  // 获取装甲板位置列表 (用于瞄准)
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // EKF相关 (兼容Target接口)
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }
  bool diverged() const;
  bool convergened() const { return state_ == OutpostState::TRACKING; }

  // 调试信息
  const OutpostModel & model() const { return model_; }
  const std::deque<OutpostObservation> & observations() const { return observations_; }

private:
  OutpostState state_ = OutpostState::LOST;
  OutpostModel model_;
  std::deque<OutpostObservation> observations_;
  std::chrono::steady_clock::time_point scan_start_time_;
  std::chrono::steady_clock::time_point last_update_time_;

  // EKF (用于平滑追踪)
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;

  // 配置参数
  double scan_timeout_ = 3.0;           // 扫描超时时间 (s)
  double layer_gap_min_ = 0.05;         // 层间最小高度差 (m)
  double layer_gap_max_ = 0.20;         // 层间最大高度差 (m)
  double cluster_threshold_ = 0.03;     // 高度聚类阈值 (m)
  double validation_error_threshold_ = 0.15;  // 验证误差阈值 (m)
  int min_observations_ = 5;            // 最少观测数
  int validation_required_ = 3;         // 需要的验证次数
  double angle_coverage_required_ = 60.0 / 57.3;  // 需要的角度覆盖 (rad)

  // 追踪状态
  int consecutive_validation_failures_ = 0;
  int temp_lost_count_ = 0;
  int max_temp_lost_count_ = 75;

  // 扫描建模
  void add_observation(const Armor & armor, std::chrono::steady_clock::time_point t);
  bool try_build_model();
  void cluster_heights();
  double calculate_angle_coverage() const;

  // 验证
  bool validate_observation(const Armor & armor, std::chrono::steady_clock::time_point t);
  int identify_layer(double z) const;

  // 追踪
  void update_tracking(const Armor & armor, std::chrono::steady_clock::time_point t);
  void init_ekf(const Armor & armor, std::chrono::steady_clock::time_point t);
  void update_ekf(const Armor & armor, int layer_id);

  // 预测
  double predict_phase(std::chrono::steady_clock::time_point t) const;
  Eigen::Vector3d predict_armor_position(double phase, int layer) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
