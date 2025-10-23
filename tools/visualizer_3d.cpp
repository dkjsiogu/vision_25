#include "visualizer_3d.hpp"

#include <fmt/core.h>

namespace tools
{

Visualizer3D::Visualizer3D(int width, int height)
: width_(width), height_(height), scale_(200.0), view_pitch_(30.0 * CV_PI / 180.0), 
  view_yaw_(45.0 * CV_PI / 180.0), mouse_dragging_(false), mouse_panning_(false), has_data_(false)
{
  center_ = cv::Point2f(width / 2.0f, height / 2.0f);
  canvas_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(20, 20, 20));
}

cv::Point2f Visualizer3D::project3D(const Eigen::Vector3d & point) const
{
  // 从观察者的角度看：相机位于球面上，看向原点
  // 使用球坐标：(r, yaw, pitch)，r是距离原点的距离（这里用固定值）
  
  // 1. 先将点从世界坐标系转换到相机坐标系
  // 相机位置（球坐标）
  double cam_r = 2.0;  // 相机距离原点的距离（固定）
  
  // 相机在世界坐标系中的位置
  double cam_x = cam_r * std::cos(view_pitch_) * std::cos(view_yaw_);
  double cam_y = cam_r * std::cos(view_pitch_) * std::sin(view_yaw_);
  double cam_z = cam_r * std::sin(view_pitch_);
  
  // 将点从世界坐标系转换到相机坐标系
  // 相机看向原点，up向量为世界Z轴方向
  Eigen::Vector3d cam_pos(cam_x, cam_y, cam_z);
  Eigen::Vector3d forward = -cam_pos.normalized();  // 看向原点
  Eigen::Vector3d world_up(0, 0, 1);
  Eigen::Vector3d right = forward.cross(world_up).normalized();
  Eigen::Vector3d up = right.cross(forward);
  
  // 构建视图矩阵（世界到相机）
  Eigen::Vector3d point_rel = point - cam_pos;
  double x_cam = point_rel.dot(right);
  double y_cam = point_rel.dot(up);
  double z_cam = point_rel.dot(forward);
  
  // 2. 透视投影到2D
  // 简单的正交投影（为了保持等轴测效果）
  float px = center_.x + x_cam * scale_;
  float py = center_.y - y_cam * scale_;  // Y轴向下为正
  
  return cv::Point2f(px, py);
}

void Visualizer3D::draw_arrow(
  const Eigen::Vector3d & from,
  const Eigen::Vector3d & to,
  const cv::Scalar & color,
  int thickness)
{
  cv::Point2f p1 = project3D(from);
  cv::Point2f p2 = project3D(to);
  
  cv::arrowedLine(canvas_, p1, p2, color, thickness, cv::LINE_AA, 0, 0.15);
}

void Visualizer3D::draw_point(
  const Eigen::Vector3d & point,
  const cv::Scalar & color,
  int radius)
{
  cv::Point2f p = project3D(point);
  cv::circle(canvas_, p, radius, color, -1, cv::LINE_AA);
  cv::circle(canvas_, p, radius + 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void Visualizer3D::draw_text_with_bg(
  const std::string & text,
  cv::Point position,
  const cv::Scalar & text_color,
  const cv::Scalar & bg_color)
{
  int baseline = 0;
  cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
  
  cv::rectangle(
    canvas_,
    cv::Point(position.x - 2, position.y - text_size.height - 2),
    cv::Point(position.x + text_size.width + 2, position.y + 2),
    bg_color, -1);
  
  cv::putText(canvas_, text, position, cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv::LINE_AA);
}

void Visualizer3D::draw_axis(
  const Eigen::Vector3d & origin,
  const Eigen::Matrix3d & rotation,
  double length,
  const std::string & label)
{
  // X轴 - 红色
  Eigen::Vector3d x_end = origin + rotation.col(0) * length;
  draw_arrow(origin, x_end, cv::Scalar(0, 0, 255), 2);
  
  // Y轴 - 绿色
  Eigen::Vector3d y_end = origin + rotation.col(1) * length;
  draw_arrow(origin, y_end, cv::Scalar(0, 255, 0), 2);
  
  // Z轴 - 蓝色
  Eigen::Vector3d z_end = origin + rotation.col(2) * length;
  draw_arrow(origin, z_end, cv::Scalar(255, 0, 0), 2);
  
  // 标签
  if (!label.empty()) {
    cv::Point2f p = project3D(origin);
    draw_text_with_bg(label, cv::Point(p.x + 5, p.y - 5), cv::Scalar(255, 255, 255));
  }
}

void Visualizer3D::draw_camera(
  const Eigen::Vector3d & position,
  const Eigen::Matrix3d & rotation)
{
  // 绘制相机原点
  draw_point(position, cv::Scalar(255, 255, 0), 6);
  
  // 绘制相机坐标轴
  draw_axis(position, rotation, 0.1, "Camera");
  
  // 绘制相机视锥体
  double cone_length = 0.15;
  double cone_width = 0.08;
  
  Eigen::Vector3d forward = position + rotation.col(2) * cone_length;
  Eigen::Vector3d top = forward + rotation.col(1) * cone_width;
  Eigen::Vector3d bottom = forward - rotation.col(1) * cone_width;
  Eigen::Vector3d left = forward - rotation.col(0) * cone_width;
  Eigen::Vector3d right = forward + rotation.col(0) * cone_width;
  
  cv::Scalar camera_color(128, 255, 255);
  cv::line(canvas_, project3D(position), project3D(top), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(position), project3D(bottom), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(position), project3D(left), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(position), project3D(right), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(top), project3D(right), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(right), project3D(bottom), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(bottom), project3D(left), camera_color, 1, cv::LINE_AA);
  cv::line(canvas_, project3D(left), project3D(top), camera_color, 1, cv::LINE_AA);
}

void Visualizer3D::draw_gimbal(
  const Eigen::Vector3d & position,
  const Eigen::Matrix3d & rotation)
{
  // 绘制云台中心
  draw_point(position, cv::Scalar(255, 128, 0), 8);
  
  // 绘制云台坐标轴
  draw_axis(position, rotation, 0.15, "Gimbal");
  
  // 绘制云台平台（简化为一个圆盘）
  cv::Point2f center = project3D(position);
  cv::circle(canvas_, center, 20, cv::Scalar(100, 100, 200), 2, cv::LINE_AA);
}

void Visualizer3D::draw_ground_grid(double size, int divisions)
{
  double step = size / divisions;
  cv::Scalar grid_color(60, 60, 60);  // 深灰色网格
  cv::Scalar axis_color(80, 80, 80);  // 稍亮的颜色表示主轴
  
  // 假设地面在Z=0平面（或者可以设置为云台下方一定距离）
  double ground_z = -0.3;  // 地面在云台下方30cm
  
  // 绘制网格线
  for (int i = -divisions; i <= divisions; i++) {
    double pos = i * step;
    
    // 沿X方向的线
    Eigen::Vector3d start_x(-size, pos, ground_z);
    Eigen::Vector3d end_x(size, pos, ground_z);
    cv::Point2f p1 = project3D(start_x);
    cv::Point2f p2 = project3D(end_x);
    cv::Scalar color = (i == 0) ? axis_color : grid_color;
    cv::line(canvas_, p1, p2, color, 1, cv::LINE_AA);
    
    // 沿Y方向的线
    Eigen::Vector3d start_y(pos, -size, ground_z);
    Eigen::Vector3d end_y(pos, size, ground_z);
    p1 = project3D(start_y);
    p2 = project3D(end_y);
    color = (i == 0) ? axis_color : grid_color;
    cv::line(canvas_, p1, p2, color, 1, cv::LINE_AA);
  }
}

void Visualizer3D::update(
  const Eigen::Matrix3d & R_camera2gimbal,
  const Eigen::Vector3d & t_camera2gimbal,
  const Eigen::Matrix3d & R_gimbal2world,
  const Eigen::Vector3d & target_xyz_in_gimbal,
  const Eigen::Vector3d & target_xyz_in_world,
  double gimbal_yaw,
  double gimbal_pitch,
  double target_yaw,
  double target_pitch)
{
  // 缓存数据
  cached_R_camera2gimbal_ = R_camera2gimbal;
  cached_t_camera2gimbal_ = t_camera2gimbal;
  cached_R_gimbal2world_ = R_gimbal2world;
  cached_target_xyz_in_gimbal_ = target_xyz_in_gimbal;
  cached_target_xyz_in_world_ = target_xyz_in_world;
  cached_gimbal_yaw_ = gimbal_yaw;
  cached_gimbal_pitch_ = gimbal_pitch;
  cached_target_yaw_ = target_yaw;
  cached_target_pitch_ = target_pitch;
  has_data_ = true;
  
  // 渲染
  render();
}

void Visualizer3D::render()
{
  if (!has_data_) return;
  
  // 清空画布
  canvas_ = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(20, 20, 20));
  
  // 定义坐标系原点
  Eigen::Vector3d gimbal_origin(0, 0, 0);
  Eigen::Vector3d world_origin(0, 0, 0);
  
  // 相机在云台坐标系中的位置
  Eigen::Vector3d camera_in_gimbal = cached_t_camera2gimbal_;
  
  // 先绘制地面网格（在底层）
  draw_ground_grid(1.0, 10);  // 1米范围，10个格子
  
  // 绘制世界坐标系
  draw_axis(world_origin, Eigen::Matrix3d::Identity(), 0.3, "World");
  
  // 绘制云台（在世界坐标系原点，带旋转）
  draw_gimbal(gimbal_origin, cached_R_gimbal2world_);
  
  // 绘制相机（在云台坐标系中的位置，转换到世界坐标系）
  Eigen::Vector3d camera_in_world = cached_R_gimbal2world_ * camera_in_gimbal;
  Eigen::Matrix3d R_camera2world = cached_R_gimbal2world_ * cached_R_camera2gimbal_;
  draw_camera(camera_in_world, R_camera2world);
  
  // 绘制相机到云台的连线
  draw_arrow(gimbal_origin, camera_in_world, cv::Scalar(255, 255, 0), 1);
  
  // 绘制目标位置
  draw_point(cached_target_xyz_in_world_, cv::Scalar(0, 255, 255), 8);
  draw_text_with_bg("Target", project3D(cached_target_xyz_in_world_) + cv::Point2f(10, 0), cv::Scalar(0, 255, 255));
  
  // 绘制云台到目标的连线
  draw_arrow(gimbal_origin, cached_target_xyz_in_world_, cv::Scalar(0, 200, 200), 1);
  
  // 绘制参数文本信息
  int text_y = 20;
  int line_height = 20;
  
  auto draw_info = [&](const std::string & text) {
    draw_text_with_bg(text, cv::Point(10, text_y), cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));
    text_y += line_height;
  };
  
  draw_info("=== Camera to Gimbal Transform ===");
  draw_info(fmt::format("Translation: [{:.3f}, {:.3f}, {:.3f}] m", 
    cached_t_camera2gimbal_.x(), cached_t_camera2gimbal_.y(), cached_t_camera2gimbal_.z()));
  draw_info(fmt::format("  Forward: {:.3f}m  Right: {:.3f}m  Up: {:.3f}m",
    cached_t_camera2gimbal_.x(), -cached_t_camera2gimbal_.y(), cached_t_camera2gimbal_.z()));
  
  text_y += 5;
  draw_info("=== Gimbal State ===");
  draw_info(fmt::format("Yaw:   {:.3f} rad / {:.2f} deg", cached_gimbal_yaw_, cached_gimbal_yaw_ * 57.3));
  draw_info(fmt::format("Pitch: {:.3f} rad / {:.2f} deg", cached_gimbal_pitch_, cached_gimbal_pitch_ * 57.3));
  
  text_y += 5;
  draw_info("=== Target Position ===");
  draw_info(fmt::format("In Gimbal: [{:.3f}, {:.3f}, {:.3f}] m",
    cached_target_xyz_in_gimbal_.x(), cached_target_xyz_in_gimbal_.y(), cached_target_xyz_in_gimbal_.z()));
  draw_info(fmt::format("In World:  [{:.3f}, {:.3f}, {:.3f}] m",
    cached_target_xyz_in_world_.x(), cached_target_xyz_in_world_.y(), cached_target_xyz_in_world_.z()));
  draw_info(fmt::format("Distance: {:.3f} m", cached_target_xyz_in_world_.norm()));
  
  text_y += 5;
  draw_info("=== Target Angles ===");
  draw_info(fmt::format("Yaw:   {:.3f} rad / {:.2f} deg", cached_target_yaw_, cached_target_yaw_ * 57.3));
  draw_info(fmt::format("Pitch: {:.3f} rad / {:.2f} deg", cached_target_pitch_, cached_target_pitch_ * 57.3));
  
  text_y += 5;
  draw_info("=== View Control ===");
  draw_info(fmt::format("View Yaw: {:.1f} deg  Pitch: {:.1f} deg", view_yaw_ * 57.3, view_pitch_ * 57.3));
  draw_info(fmt::format("Scale: {:.0f}  Center: [{:.0f}, {:.0f}]", scale_, center_.x, center_.y));
  
  // 绘制坐标轴图例
  int legend_x = width_ - 150;
  int legend_y = height_ - 100;
  draw_text_with_bg("Axes Legend:", cv::Point(legend_x, legend_y), cv::Scalar(255, 255, 255));
  cv::line(canvas_, cv::Point(legend_x, legend_y + 15), cv::Point(legend_x + 30, legend_y + 15), cv::Scalar(0, 0, 255), 2);
  draw_text_with_bg("X", cv::Point(legend_x + 35, legend_y + 20), cv::Scalar(0, 0, 255));
  cv::line(canvas_, cv::Point(legend_x, legend_y + 30), cv::Point(legend_x + 30, legend_y + 30), cv::Scalar(0, 255, 0), 2);
  draw_text_with_bg("Y", cv::Point(legend_x + 35, legend_y + 35), cv::Scalar(0, 255, 0));
  cv::line(canvas_, cv::Point(legend_x, legend_y + 45), cv::Point(legend_x + 30, legend_y + 45), cv::Scalar(255, 0, 0), 2);
  draw_text_with_bg("Z", cv::Point(legend_x + 35, legend_y + 50), cv::Scalar(255, 0, 0));
  
  // 添加操作提示
  draw_text_with_bg("Controls: Left-drag=Rotate | Right-drag=Pan | Scroll=Zoom", 
    cv::Point(10, height_ - 10), cv::Scalar(150, 150, 150));
}

void Visualizer3D::mouse_callback(int event, int x, int y, int flags, void* userdata)
{
  Visualizer3D* vis = static_cast<Visualizer3D*>(userdata);
  
  if (event == cv::EVENT_LBUTTONDOWN) {
    // 左键按下 - 开始旋转
    vis->mouse_dragging_ = true;
    vis->mouse_panning_ = false;
    vis->last_mouse_pos_ = cv::Point2i(x, y);
  }
  else if (event == cv::EVENT_RBUTTONDOWN) {
    // 右键按下 - 开始平移
    vis->mouse_dragging_ = true;
    vis->mouse_panning_ = true;
    vis->last_mouse_pos_ = cv::Point2i(x, y);
  }
  else if (event == cv::EVENT_LBUTTONUP || event == cv::EVENT_RBUTTONUP) {
    // 鼠标释放
    vis->mouse_dragging_ = false;
  }
  else if (event == cv::EVENT_MOUSEMOVE && vis->mouse_dragging_) {
    int dx = x - vis->last_mouse_pos_.x;
    int dy = y - vis->last_mouse_pos_.y;
    vis->last_mouse_pos_ = cv::Point2i(x, y);
    
    if (vis->mouse_panning_) {
      // 平移画面
      vis->center_.x += dx;
      vis->center_.y += dy;
    } else {
      // 旋转视角（像轨迹球一样）
      vis->view_yaw_ -= dx * 0.005;  // 左右拖动改变yaw（注意负号：向右拖动逆时针）
      vis->view_pitch_ += dy * 0.005;  // 上下拖动改变pitch
      
      // 限制pitch范围（避免翻转）
      const double max_pitch = 85.0 * CV_PI / 180.0;  // 最大85度
      if (vis->view_pitch_ < -max_pitch) vis->view_pitch_ = -max_pitch;
      if (vis->view_pitch_ > max_pitch) vis->view_pitch_ = max_pitch;
    }
    
    // 重新渲染
    vis->render();
    cv::imshow("3D Coordinate System", vis->canvas_);
  }
  else if (event == cv::EVENT_MOUSEWHEEL) {
    // 滚轮缩放
    int delta = cv::getMouseWheelDelta(flags);
    if (delta > 0) {
      vis->scale_ *= 1.1;
    } else {
      vis->scale_ *= 0.9;
    }
    
    // 限制缩放范围
    if (vis->scale_ < 50.0) vis->scale_ = 50.0;
    if (vis->scale_ > 1000.0) vis->scale_ = 1000.0;
    
    // 重新渲染
    vis->render();
    cv::imshow("3D Coordinate System", vis->canvas_);
  }
}

void Visualizer3D::show(const std::string & window_name)
{
  cv::imshow(window_name, canvas_);
  
  // 设置鼠标回调（只设置一次）
  static bool callback_set = false;
  if (!callback_set) {
    cv::setMouseCallback(window_name, mouse_callback, this);
    callback_set = true;
  }
}

}  // namespace tools
