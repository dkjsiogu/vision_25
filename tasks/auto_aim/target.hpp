#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, std::vector<double> height_offsets = {});
  Target(double x, double vyaw, double radius, double h);

  // 从外部数据构造 (用于OutpostTarget适配)
  Target(
    ArmorName name, ArmorType type, ArmorPriority priority, bool jumped, int last_id,
    const Eigen::VectorXd & ekf_x, const std::vector<Eigen::Vector4d> & armor_list, int armor_num);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  // 动态高度偏移学习（支持不同高度的装甲板，如前哨站三层）
  std::vector<double> height_offset_;           // 各装甲板相对于基准z的高度偏移
  std::vector<bool> height_offset_initialized_; // 是否已初始化
  bool use_dynamic_height_ = false;             // 是否使用动态高度学习

  std::vector<Eigen::Vector4d> external_armor_list_;  // 外部提供的装甲板列表
  bool use_external_armor_list_ = false;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  void update_height_offset(const Armor & armor, int id);  // 更新高度偏移

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP