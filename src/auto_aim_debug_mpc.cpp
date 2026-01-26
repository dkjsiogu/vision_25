#include <fmt/core.h>
#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/debug_recorder.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/timing_validator.hpp"
#include "tools/visualizer_3d.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Visualizer3D visualizer(1200, 800);

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  // 启用前哨站调试日志
  tools::DebugRecorder::instance("outpost").enable("outpost_log.csv");

  // 读取 timing 验证开关（默认开启）
  auto yaml = YAML::LoadFile(config_path);
  bool timing_validation_enabled = true;
  if (yaml["timing_validation_enabled"]) {
    timing_validation_enabled = yaml["timing_validation_enabled"].as<bool>();
  }
  tools::TimingValidator timing_validator;
  if (timing_validation_enabled) {
    timing_validator.enable("timing_validation.csv");
    tools::logger()->info("[TimingValidator] Enabled, output: timing_validation.csv");
  }

  // 存储 q 和时间戳用于 timing 验证
  Eigen::Quaterniond last_q = Eigen::Quaterniond::Identity();
  std::chrono::steady_clock::time_point last_imu_t;

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<double> shared_fps{0.0};  // 共享 FPS 变量

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;
    double plan_fps = 0.0;
    int plan_count = 0;
    auto plan_fps_start = std::chrono::steady_clock::now();

    while (!quit) {
      auto plan_loop_start = std::chrono::steady_clock::now();

      // 计算规划器频率
      plan_count++;
      auto plan_fps_elapsed =
        std::chrono::duration<double>(plan_loop_start - plan_fps_start).count();
      if (plan_fps_elapsed >= 1.0) {
        plan_fps = plan_count / plan_fps_elapsed;
        plan_count = 0;
        plan_fps_start = plan_loop_start;
      }

      auto target = target_queue.front();
      auto gs = gimbal.state();
      // [改进] 使用云台实测状态作为 MPC 初值
      auto plan = planner.plan(target, gs.bullet_speed, gs);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      // 计算控制响应速度（跟踪误差）
      double yaw_error = plan.control ? (plan.target_yaw - gs.yaw) : 0.0;
      double pitch_error = plan.control ? (plan.target_pitch - gs.pitch) : 0.0;
      double plan_yaw_error = plan.control ? (plan.yaw - gs.yaw) : 0.0;
      double plan_pitch_error = plan.control ? (plan.pitch - gs.pitch) : 0.0;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;
      const double rad2deg = 57.29577951308232;
      auto target_pitch_str = plan.control && target.has_value()
                                ? fmt::format("{:.3f} rad/{:.3f} deg", plan.target_pitch,
                                               plan.target_pitch * rad2deg)
                                : "N/A";
      auto target_yaw_str = plan.control && target.has_value()
                               ? fmt::format("{:.3f} rad/{:.3f} deg", plan.target_yaw,
                                              plan.target_yaw * rad2deg)
                               : "N/A";
      tools::logger()->info(
        "[Angles] gimbal yaw {:.3f} rad/{:.3f} deg pitch {:.3f} rad/{:.3f} deg | target yaw {} "
        "pitch {} | plan yaw {:.3f} rad/{:.3f} deg pitch {:.3f} rad/{:.3f} deg",
        gs.yaw, gs.yaw * rad2deg, gs.pitch, gs.pitch * rad2deg, target_yaw_str, target_pitch_str,
        plan.yaw, plan.yaw * rad2deg, plan.pitch, plan.pitch * rad2deg);

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      // 添加 FPS 和控制响应速度数据
      data["fps"] = shared_fps.load();
      data["plan_fps"] = plan_fps;
      data["yaw_error"] = yaw_error;
      data["pitch_error"] = pitch_error;
      data["plan_yaw_error"] = plan_yaw_error;
      data["plan_pitch_error"] = plan_pitch_error;

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  // FPS 计算相关变量
  auto fps_start_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  double current_fps = 0.0;

  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, t);
    auto q = gimbal.q(t);
    auto gs = gimbal.state();

    // 计算 FPS
    frame_count++;
    auto fps_elapsed = std::chrono::duration<double>(frame_start - fps_start_time).count();
    if (fps_elapsed >= 1.0) {
      current_fps = frame_count / fps_elapsed;
      frame_count = 0;
      fps_start_time = frame_start;
      shared_fps.store(current_fps);  // 更新共享变量供 plan_thread 使用
    }

    // 记录当前帧的 q（用于 timing 验证）
    last_q = q;
    last_imu_t = t;  // gimbal.q(t) 会插值到这个时间点

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);

    // Timing 验证：对每个检测到的 armor，记录原始观测数据
    if (timing_validator.is_enabled() && !armors.empty()) {
      for (auto & armor : armors) {
        // 调用 solve 获取 xyz_in_gimbal, xyz_in_world
        solver.solve(armor);

        timing_validator.record(
          t,                          // cam_t
          last_imu_t,                 // imu_t (插值目标时间)
          gs.yaw, gs.pitch,           // 云台角度
          last_q,                     // IMU 四元数
          armor.xyz_in_gimbal,        // 云台系坐标
          armor.xyz_in_world,         // "世界"系坐标
          armor.ypr_in_world[0],      // armor yaw
          armor.ypr_in_world[1],      // armor pitch
          armor.ypd_in_world[2],      // 距离
          auto_aim::ARMOR_NAMES[armor.name]
        );
      }
    }

    auto targets = tracker.track(armors, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
      
      // 更新3D可视化
      auto plan = planner.plan(std::make_optional(target), gs.bullet_speed);
      Eigen::Vector3d target_xyz_in_world = aim_xyza.head(3);
      Eigen::Vector3d target_xyz_in_gimbal = solver.R_gimbal2world().transpose() * target_xyz_in_world;
      
      visualizer.update(
        solver.R_camera2gimbal(),
        solver.t_camera2gimbal(),
        solver.R_gimbal2world(),
        target_xyz_in_gimbal,
        target_xyz_in_world,
        gs.yaw,
        gs.pitch,
        plan.target_yaw,
        plan.target_pitch
      );
      visualizer.show("3D Coordinate System");
    }

    // 在图像上显示 FPS 和状态信息
    cv::putText(
      img, fmt::format("FPS: {:.1f}", current_fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 0), 2);
    cv::putText(
      img, fmt::format("Gimbal: yaw={:.2f} pitch={:.2f}", gs.yaw, gs.pitch), cv::Point(10, 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}