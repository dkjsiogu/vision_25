#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>

namespace tools
{

class Visualizer3D
{
public:
  Visualizer3D(int width = 800, int height = 600);

  // 更新可视化数据
  void update(
    const Eigen::Matrix3d & R_camera2gimbal,
    const Eigen::Vector3d & t_camera2gimbal,
    const Eigen::Matrix3d & R_gimbal2world,
    const Eigen::Vector3d & target_xyz_in_gimbal,
    const Eigen::Vector3d & target_xyz_in_world,
    double gimbal_yaw,
    double gimbal_pitch,
    double target_yaw,
    double target_pitch);

  // 显示可视化窗口
  void show(const std::string & window_name = "3D Visualization");

  // 获取可视化图像
  cv::Mat get_image() const { return canvas_.clone(); }
  
  // 鼠标回调函数（静态）
  static void mouse_callback(int event, int x, int y, int flags, void* userdata);

private:
  int width_;
  int height_;
  cv::Mat canvas_;
  
  double scale_;          // 缩放比例 (像素/米)
  cv::Point2f center_;    // 画布中心点
  double view_pitch_;     // 观察俯仰角
  double view_yaw_;       // 观察偏航角
  
  // 鼠标交互状态
  bool mouse_dragging_;
  cv::Point2i last_mouse_pos_;
  bool mouse_panning_;
  
  // 缓存的数据（用于重绘）
  Eigen::Matrix3d cached_R_camera2gimbal_;
  Eigen::Vector3d cached_t_camera2gimbal_;
  Eigen::Matrix3d cached_R_gimbal2world_;
  Eigen::Vector3d cached_target_xyz_in_gimbal_;
  Eigen::Vector3d cached_target_xyz_in_world_;
  double cached_gimbal_yaw_;
  double cached_gimbal_pitch_;
  double cached_target_yaw_;
  double cached_target_pitch_;
  bool has_data_;

  // 3D到2D投影 (简单的等轴测投影)
  cv::Point2f project3D(const Eigen::Vector3d & point) const;
  
  // 绘制坐标轴
  void draw_axis(
    const Eigen::Vector3d & origin,
    const Eigen::Matrix3d & rotation,
    double length,
    const std::string & label);
  
  // 绘制向量箭头
  void draw_arrow(
    const Eigen::Vector3d & from,
    const Eigen::Vector3d & to,
    const cv::Scalar & color,
    int thickness = 2);
  
  // 绘制点
  void draw_point(
    const Eigen::Vector3d & point,
    const cv::Scalar & color,
    int radius = 5);
  
  // 绘制文本（带背景）
  void draw_text_with_bg(
    const std::string & text,
    cv::Point position,
    const cv::Scalar & text_color,
    const cv::Scalar & bg_color = cv::Scalar(0, 0, 0));
  
  // 绘制相机模型
  void draw_camera(
    const Eigen::Vector3d & position,
    const Eigen::Matrix3d & rotation);
  
  // 绘制云台模型
  void draw_gimbal(
    const Eigen::Vector3d & position,
    const Eigen::Matrix3d & rotation);
  
  // 绘制地面网格
  void draw_ground_grid(double size, int divisions);
  
  // 渲染场景（使用缓存的数据）
  void render();
};

}  // namespace tools
