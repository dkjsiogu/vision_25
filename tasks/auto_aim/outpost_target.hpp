#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
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
 * - 上中下三层，各层高度差0.1m
 * - 俯视图120度分布
 * - 同一个旋转中心、角速度、半径
 *
 * 状态向量 (8维):
 * [center_x, vx, center_y, vy, z, phase0, omega, radius]
 *
 * 三个装甲板位置 (角度匹配确定id):
 * - id=0: phase = phase0, z = z + height_offset[0]
 * - id=1: phase = phase0 + 120°, z = z + height_offset[1]
 * - id=2: phase = phase0 + 240°, z = z + height_offset[2]
 *
 * 关键设计：
 * 1. 用角度匹配确定观测到的是哪个装甲板(id)
 * 2. z在EKF状态中，pitch观测可以修正z
 * 3. 不同装甲板的高度差通过height_offset记录
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

  int current_id() const { return current_id_; }

  // 获取高度偏移列表 (用于传递给Target)
  std::vector<double> height_offsets() const
  {
    std::vector<double> offsets;
    for (int i = 0; i < 3; i++) {
      offsets.push_back(height_offset_initialized_[i] ? height_offset_[i] : 0.0);
    }
    return offsets;
  }

private:
  OutpostState state_ = OutpostState::LOST;

  // EKF: [cx, vx, cy, vy, z, phase0, omega, radius]
  tools::ExtendedKalmanFilter ekf_;
  bool ekf_initialized_ = false;
  int update_count_ = 0;

  // 三个装甲板相对于基准z的高度偏移 (id=0,1,2)
  double height_offset_[3] = {0, 0, 0};
  bool height_offset_initialized_[3] = {false, false, false};

  int current_id_ = -1;
  std::chrono::steady_clock::time_point last_update_time_;

  int max_temp_lost_count_ = 75;
  int temp_lost_count_ = 0;
  double outpost_radius_ = 0.2765;

  void init_ekf(const Armor & armor);

  // 通过角度匹配确定观测到的是哪个装甲板 (返回0/1/2)
  int match_armor_id(const Armor & armor) const;

  // 更新指定id的装甲板
  void update_armor(const Armor & armor, int id);

  // 计算第id个装甲板的位置
  Eigen::Vector4d armor_xyza(int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
