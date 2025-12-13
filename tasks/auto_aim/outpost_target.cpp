#include "outpost_target.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

OutpostTarget::OutpostTarget(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  if (yaml["outpost_max_temp_lost_count"]) {
    max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  }
  if (yaml["outpost_radius"]) {
    outpost_radius_ = yaml["outpost_radius"].as<double>();
  }

  reset();
}

std::string OutpostTarget::state_string() const
{
  switch (state_) {
    case OutpostState::LOST:
      return "lost";
    case OutpostState::TRACKING:
      return "tracking";
    default:
      return "unknown";
  }
}

void OutpostTarget::reset()
{
  state_ = OutpostState::LOST;
  ekf_initialized_ = false;
  update_count_ = 0;
  current_id_ = -1;
  temp_lost_count_ = 0;
  jumped = false;
  last_id = 0;

  for (int i = 0; i < 3; i++) {
    armor_z_[i] = 0;
    armor_z_initialized_[i] = false;
  }
}

void OutpostTarget::init_ekf(const Armor & armor)
{
  double armor_yaw = armor.ypr_in_world[0];
  double armor_x = armor.xyz_in_world[0];
  double armor_y = armor.xyz_in_world[1];

  // 从装甲板位置推算旋转中心
  double cx = armor_x + outpost_radius_ * std::cos(armor_yaw);
  double cy = armor_y + outpost_radius_ * std::sin(armor_yaw);

  // 状态: [cx, vx, cy, vy, phase0, omega, radius]
  // 初始化时，假设观测到的是id=0的装甲板，所以phase0 = armor_yaw
  Eigen::VectorXd x0(7);
  x0 << cx, 0, cy, 0, armor_yaw, 0, outpost_radius_;

  Eigen::VectorXd P0_dig(7);
  P0_dig << 0.5, 16, 0.5, 16, 0.4, 1, 1e-4;
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[4] = tools::limit_rad(c[4]);  // phase0
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_initialized_ = true;

  // 初始化id=0的z
  armor_z_[0] = armor.xyz_in_world[2];
  armor_z_initialized_[0] = true;
  current_id_ = 0;

  tools::logger()->info(
    "[OutpostTarget] Init EKF: cx={:.3f}, cy={:.3f}, phase0={:.3f}, z0={:.3f}", cx, cy, armor_yaw,
    armor_z_[0]);
}

int OutpostTarget::match_armor_id(const Armor & armor) const
{
  if (!ekf_initialized_) return 0;

  double observed_yaw = armor.ypr_in_world[0];
  double phase0 = ekf_.x[4];

  double min_error = 1e10;
  int best_id = 0;

  for (int id = 0; id < 3; id++) {
    double predicted_phase = tools::limit_rad(phase0 + id * 2 * M_PI / 3);
    double error = std::abs(tools::limit_rad(observed_yaw - predicted_phase));

    if (error < min_error) {
      min_error = error;
      best_id = id;
    }
  }

  return best_id;
}

void OutpostTarget::update_armor(const Armor & armor, int id)
{
  // 更新该id的z (滑动平均)
  if (!armor_z_initialized_[id]) {
    armor_z_[id] = armor.xyz_in_world[2];
    armor_z_initialized_[id] = true;
  } else {
    double alpha = 0.1;
    armor_z_[id] = armor_z_[id] * (1 - alpha) + armor.xyz_in_world[2] * alpha;
  }

  // 该id的phase偏移
  double phase_offset = id * 2 * M_PI / 3;

  // 观测量: [yaw, pitch, distance, armor_yaw]
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z_obs(4);
  z_obs << ypd[0], ypd[1], ypd[2], ypr[0];

  // 计算雅可比矩阵
  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];
  double phase_id = tools::limit_rad(phase0 + phase_offset);

  double armor_x = cx - r * std::cos(phase_id);
  double armor_y = cy - r * std::sin(phase_id);
  Eigen::Vector3d armor_xyz(armor_x, armor_y, armor_z_[id]);

  // H_xyz_state: 装甲板xyz对状态的雅可比
  // armor_x = cx - r * cos(phase0 + offset)
  // armor_y = cy - r * sin(phase0 + offset)
  // d(armor_x)/d(cx) = 1, d(armor_x)/d(phase0) = r * sin(phase_id), d(armor_x)/d(r) = -cos(phase_id)
  // d(armor_y)/d(cy) = 1, d(armor_y)/d(phase0) = -r * cos(phase_id), d(armor_y)/d(r) = -sin(phase_id)
  // clang-format off
  Eigen::MatrixXd H_xyz_state(3, 7);
  H_xyz_state <<
    1, 0, 0, 0,  r * std::sin(phase_id), 0, -std::cos(phase_id),
    0, 0, 1, 0, -r * std::cos(phase_id), 0, -std::sin(phase_id),
    0, 0, 0, 0,                       0, 0,                   0;
  // clang-format on

  Eigen::MatrixXd H_ypd_xyz = tools::xyz2ypd_jacobian(armor_xyz);

  // 完整的观测雅可比 (4x7): [yaw, pitch, distance, armor_yaw] 对 [cx, vx, cy, vy, phase0, omega, r]
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 7);
  H.block<3, 7>(0, 0) = H_ypd_xyz * H_xyz_state;
  // d(armor_yaw)/d(phase0) = 1 (因为 armor_yaw = phase0 + offset)
  H(3, 4) = 1;

  // 观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig(4);
  R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
    std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  auto h_func = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    double cx_ = x[0], cy_ = x[2];
    double phase0_ = x[4], r_ = x[6];
    double phase_id_ = tools::limit_rad(phase0_ + phase_offset);

    double ax = cx_ - r_ * std::cos(phase_id_);
    double ay = cy_ - r_ * std::sin(phase_id_);

    Eigen::Vector3d xyz(ax, ay, armor_z_[id]);
    Eigen::Vector3d ypd_pred = tools::xyz2ypd(xyz);

    return {ypd_pred[0], ypd_pred[1], ypd_pred[2], phase_id_};
  };

  ekf_.update(z_obs, H, R, h_func, z_subtract);
  update_count_++;
}

bool OutpostTarget::update(const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (armor.name != ArmorName::outpost) {
    return false;
  }

  last_update_time_ = t;
  temp_lost_count_ = 0;
  priority = armor.priority;

  if (state_ == OutpostState::LOST) {
    // 第一次识别
    init_ekf(armor);
    current_id_ = 0;
    state_ = OutpostState::TRACKING;
    return true;
  }

  // TRACKING状态: 用角度匹配确定id
  int id = match_armor_id(armor);

  if (id != current_id_) {
    jumped = true;
  }
  current_id_ = id;
  last_id = id;

  update_armor(armor, id);

  return true;
}

void OutpostTarget::predict(std::chrono::steady_clock::time_point t)
{
  if (state_ != OutpostState::TRACKING || !ekf_initialized_) {
    return;
  }

  double dt = tools::delta_time(t, last_update_time_);
  if (dt <= 0 || dt > 0.1) return;

  predict(dt);
  last_update_time_ = t;
  temp_lost_count_++;

  if (temp_lost_count_ > max_temp_lost_count_) {
    tools::logger()->info("[OutpostTarget] Lost (temp_lost_count > {})", max_temp_lost_count_);
    reset();
  }
}

void OutpostTarget::predict(double dt)
{
  if (!ekf_initialized_ || dt <= 0) return;

  // 状态转移矩阵: [cx, vx, cy, vy, phase0, omega, radius]
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0, 0},
    {0,  1,  0,  0,  0,  0, 0},
    {0,  0,  1, dt,  0,  0, 0},
    {0,  0,  0,  1,  0,  0, 0},
    {0,  0,  0,  0,  1, dt, 0},
    {0,  0,  0,  0,  0,  1, 0},
    {0,  0,  0,  0,  0,  0, 1}
  };
  // clang-format on

  double v1 = 10;   // 位置加速度方差
  double v2 = 0.1;  // 角加速度方差
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0, 0},
    {b * v1, c * v1,      0,      0,      0,      0, 0},
    {     0,      0, a * v1, b * v1,      0,      0, 0},
    {     0,      0, b * v1, c * v1,      0,      0, 0},
    {     0,      0,      0,      0, a * v2, b * v2, 0},
    {     0,      0,      0,      0, b * v2, c * v2, 0},
    {     0,      0,      0,      0,      0,      0, 0}
  };
  // clang-format on

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[4] = tools::limit_rad(x_prior[4]);
    return x_prior;
  };

  // 限制角速度范围
  if (update_count_ > 10 && std::abs(ekf_.x[5]) > 2) {
    ekf_.x[5] = ekf_.x[5] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

Eigen::Vector4d OutpostTarget::armor_xyza(int id) const
{
  if (!ekf_initialized_) return {0, 0, 0, 0};

  double cx = ekf_.x[0], cy = ekf_.x[2];
  double phase0 = ekf_.x[4], r = ekf_.x[6];

  double phase_id = tools::limit_rad(phase0 + id * 2 * M_PI / 3);

  double armor_x = cx - r * std::cos(phase_id);
  double armor_y = cy - r * std::sin(phase_id);

  // 使用该id对应的z，如果未初始化则用已初始化的z的平均值
  double z = 0;
  if (armor_z_initialized_[id]) {
    z = armor_z_[id];
  } else {
    int count = 0;
    for (int i = 0; i < 3; i++) {
      if (armor_z_initialized_[i]) {
        z += armor_z_[i];
        count++;
      }
    }
    if (count > 0) z /= count;
  }

  return {armor_x, armor_y, z, phase_id};
}

std::vector<Eigen::Vector4d> OutpostTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> list;

  if (!ekf_initialized_) {
    return list;
  }

  // 返回所有已初始化z的装甲板
  for (int i = 0; i < 3; i++) {
    if (armor_z_initialized_[i]) {
      list.push_back(armor_xyza(i));
    }
  }

  // 如果只有一个装甲板被初始化，也输出它（用于初期追踪）
  if (list.empty() && ekf_initialized_) {
    list.push_back(armor_xyza(0));
  }

  return list;
}

Eigen::VectorXd OutpostTarget::ekf_x() const
{
  Eigen::VectorXd x(11);

  if (!ekf_initialized_) {
    x << 0, 0, 0, 0, 0, 0, 0, 0, outpost_radius_, 0, 0;
    return x;
  }

  // 选择一个z值
  double z = 0;
  if (current_id_ >= 0 && armor_z_initialized_[current_id_]) {
    z = armor_z_[current_id_];
  } else {
    for (int i = 0; i < 3; i++) {
      if (armor_z_initialized_[i]) {
        z = armor_z_[i];
        break;
      }
    }
  }

  // 转换为Target兼容的11维状态: [cx, vx, cy, vy, z, vz, phase, omega, radius, l, h]
  x << ekf_.x[0], ekf_.x[1], ekf_.x[2], ekf_.x[3], z, 0, ekf_.x[4], ekf_.x[5], ekf_.x[6], 0, 0;
  return x;
}

bool OutpostTarget::diverged() const
{
  if (!ekf_initialized_) return false;

  double r = ekf_.x[6];
  if (r < 0.1 || r > 0.5) return true;

  double omega = std::abs(ekf_.x[5]);
  if (omega > 5.0) return true;

  return false;
}

bool OutpostTarget::convergened() const
{
  if (!ekf_initialized_) return false;
  return update_count_ > 10;
}

}  // namespace auto_aim
