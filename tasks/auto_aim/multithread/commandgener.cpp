#include "commandgener.hpp"

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace multithread
{

CommandGener::CommandGener(
  auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
  tools::Plotter & plotter, bool debug)
: shooter_(shooter), aimer_(aimer), cboard_(cboard), plotter_(plotter), debug_(debug)
{
  thread_ = std::thread(&CommandGener::generate_command, this);
}

CommandGener::~CommandGener()
{
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void CommandGener::push(
  const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
  double bullet_speed, const Eigen::Vector3d & gimbal_pos)
{
  {
    std::lock_guard<std::mutex> lock(mtx_);
    latest_ = {targets, t, bullet_speed, gimbal_pos};
  }
  cv_.notify_one();
}

void CommandGener::generate_command()
{
  auto t0 = std::chrono::steady_clock::now();

  io::Command last_sent_cmd{false, false, 0, 0};
  bool has_last_sent_cmd = false;
  bool last_control = false;

  while (!stop_.load(std::memory_order_acquire)) {
    std::optional<Input> input;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait_for(lock, std::chrono::milliseconds(10), [&] {
        return stop_.load(std::memory_order_acquire) || latest_.has_value();
      });

      if (stop_.load(std::memory_order_acquire)) return;

      if (latest_ && tools::delta_time(std::chrono::steady_clock::now(), latest_->t) < 0.2) {
        input = latest_;
      } else {
        input = std::nullopt;
      }
    }
    if (input) {
      auto command = aimer_.aim(input->targets_, input->t, input->bullet_speed);
      command.shoot = shooter_.shoot(command, aimer_, input->targets_, input->gimbal_pos);
      command.horizon_distance = input->targets_.empty()
                                   ? 0
                                   : std::sqrt(
                                       tools::square(input->targets_.front().ekf_x()[0]) +
                                       tools::square(input->targets_.front().ekf_x()[2]));

      if (command.control) {
        cboard_.send(command);
        last_sent_cmd = command;
        has_last_sent_cmd = true;
        last_control = true;
      } else {
        // 避免 control=false 时下位机仍读取 yaw/pitch 导致跳转到 0。
        if (last_control && has_last_sent_cmd) {
          command.shoot = false;
          command.yaw = last_sent_cmd.yaw;
          command.pitch = last_sent_cmd.pitch;
          cboard_.send(command);
        }
        last_control = false;
      }

      if (debug_) {
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["cmd_yaw"] = command.yaw * 57.3;
        data["cmd_pitch"] = command.pitch * 57.3;
        data["shoot"] = command.shoot;
        data["horizon_distance"] = command.horizon_distance;
        plotter_.plot(data);
      }
    }
  }
}

}  // namespace multithread

}  // namespace auto_aim