#ifndef TOOLS__TIMING_VALIDATOR_HPP
#define TOOLS__TIMING_VALIDATOR_HPP

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

namespace tools
{

/**
 * Timing 与坐标系验证工具
 *
 * 用于记录原始数据，验证：
 * 1. IMU 姿态补偿是否正确（云台转动时 xyz_in_world 应稳定）
 * 2. 时间同步是否合理（可尝试不同 offset）
 * 3. 标定误差的视角依赖性
 *
 * 输出 CSV 格式，可用 Python 分析：
 *   - 固定目标：xyz_in_world 应不随云台角度变化
 *   - 前哨站：旋转中心应稳定
 *
 * 使用方法：
 *   在 yaml 中设置 timing_validation_enabled: true
 *   输出文件：timing_validation.csv
 */
class TimingValidator
{
public:
  TimingValidator() = default;

  void enable(const std::string & path)
  {
    if (file_.is_open()) file_.close();
    file_.open(path);
    enabled_ = file_.is_open();
    if (enabled_) {
      // CSV header - 关键差值列便于直接分析
      file_ << "frame,t_ms,"
            << "dt_cam_imu_us,"                 // 相机-IMU时间差 (微秒)，核心指标
            << "gimbal_yaw,gimbal_pitch,"       // 云台角度 (rad)
            << "q_w,q_x,q_y,q_z,"               // IMU 四元数
            << "x_gimbal,y_gimbal,z_gimbal,"    // 云台系坐标
            << "x_world,y_world,z_world,"       // "世界"系坐标
            << "armor_yaw,armor_pitch,"         // 装甲板姿态 (world系)
            << "dist,"                          // 距离
            << "target_name"
            << "\n";
      start_time_ = std::chrono::steady_clock::now();
      frame_ = 0;
    }
  }

  void disable()
  {
    if (file_.is_open()) file_.close();
    enabled_ = false;
  }

  bool is_enabled() const { return enabled_; }

  // 记录一帧完整数据
  // cam_t: 相机图像时间戳
  // imu_t: 用于获取 q 的时间戳（通常就是 cam_t，但 Gimbal::q() 内部会插值）
  void record(
    std::chrono::steady_clock::time_point cam_t,
    std::chrono::steady_clock::time_point imu_t,
    double gimbal_yaw, double gimbal_pitch,
    const Eigen::Quaterniond & q,
    const Eigen::Vector3d & xyz_gimbal,
    const Eigen::Vector3d & xyz_world,
    double armor_yaw, double armor_pitch,
    double dist,
    const std::string & target_name)
  {
    if (!enabled_) return;

    frame_++;
    auto now = std::chrono::steady_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(now - start_time_).count();

    // 关键：相机与 IMU 时间差（微秒）
    // 正值 = 相机时间戳晚于 IMU 时间戳
    auto dt_cam_imu_us = std::chrono::duration_cast<std::chrono::microseconds>(
      cam_t - imu_t).count();

    file_ << std::fixed << std::setprecision(6);
    file_ << frame_ << "," << t_ms << ","
          << dt_cam_imu_us << ","
          << gimbal_yaw << "," << gimbal_pitch << ","
          << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
          << xyz_gimbal[0] << "," << xyz_gimbal[1] << "," << xyz_gimbal[2] << ","
          << xyz_world[0] << "," << xyz_world[1] << "," << xyz_world[2] << ","
          << armor_yaw << "," << armor_pitch << ","
          << dist << ","
          << target_name
          << "\n";
    file_.flush();
  }

private:
  bool enabled_ = false;
  std::ofstream file_;
  std::chrono::steady_clock::time_point start_time_;
  int frame_ = 0;
};

}  // namespace tools

#endif  // TOOLS__TIMING_VALIDATOR_HPP
