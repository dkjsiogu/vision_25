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
    const Eigen::VectorXd & ekf_x, const std::vector<Eigen::Vector4d> & armor_list, int armor_num,
    std::chrono::steady_clock::time_point t, std::vector<double> height_offsets = {},
    std::vector<int> initialized_ids = {});

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

  std::vector<double> height_offsets_;  // 前哨站各层高度偏移
  std::vector<int> initialized_ids_;    // 已初始化的装甲板ID列表
  std::vector<Eigen::Vector4d> external_armor_list_;  // 外部提供的装甲板列表
  bool use_external_armor_list_ = false;

  double observed_z_ = 0.0;       // 前哨站：最近观测到的 z 值
  bool observed_z_valid_ = false; // 观测 z 是否有效

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP