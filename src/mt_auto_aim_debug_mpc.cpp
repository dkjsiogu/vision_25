#include <fmt/core.h>
#include <fmt/format.h>

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
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/visualizer_3d.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml   | 位置参数，yaml配置文件路径 }"
  "{num-threads    | 4                     | YOLO线程数 }";

// 带调试信息的帧结构
struct DebugFrame
{
  int id;
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Target> target;
  
  // 调试信息
  Eigen::Vector4d aim_xyza;
  
  // 性能分析时间戳
  std::chrono::steady_clock::time_point camera_time;
  std::chrono::steady_clock::time_point detect_start;
  std::chrono::steady_clock::time_point detect_end;
  std::chrono::steady_clock::time_point track_end;
};

tools::ThreadSafeQueue<DebugFrame, true> detection_queue(2);
tools::ThreadSafeQueue<DebugFrame, true> visualization_queue(2);
tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Visualizer3D visualizer(1200, 800);

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  auto num_threads = cli.get<int>("num-threads");
  
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::logger()->info("Starting multi-threaded auto aim MPC DEBUG with {} YOLO threads", num_threads);

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  // 创建YOLO线程池（debug模式）
  tools::YOLOThreadPool yolo_pool(config_path, num_threads, true);

  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  int frame_id = 0;

  // ========== 线程1: 相机采集 + YOLO检测线程 ==========
  auto capture_thread = std::thread([&]() {
    tools::logger()->info("[Capture] Thread started");
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit) {
      auto camera_time = std::chrono::steady_clock::now();
      camera.read(img, t);
      auto q = gimbal.q(t);

      frame_id++;
      
      tools::Frame yolo_frame;
      yolo_frame.id = frame_id;
      yolo_frame.img = img.clone();  // debug模式保留原图用于显示
      yolo_frame.t = t;
      yolo_frame.q = q;

      // 提交到YOLO线程池
      yolo_pool.detect_async(std::move(yolo_frame), [frame_id, camera_time, t, q, img = img.clone()](tools::Frame && yolo_frame) mutable {
        DebugFrame processed_frame;
        processed_frame.id = frame_id;
        processed_frame.img = std::move(img);
        processed_frame.t = t;
        processed_frame.q = q;
        processed_frame.armors = std::move(yolo_frame.armors);
        processed_frame.camera_time = camera_time;
        processed_frame.detect_start = camera_time;
        processed_frame.detect_end = std::chrono::steady_clock::now();
        detection_queue.push(std::move(processed_frame));
      });
    }
    tools::logger()->info("[Capture] Thread stopped");
  });

  // ========== 线程2: 追踪线程 ==========
  auto track_thread = std::thread([&]() {
    tools::logger()->info("[Track] Thread started");

    while (!quit) {
      auto frame = detection_queue.pop();
      
      solver.set_R_gimbal2world(frame.q);
      auto targets = tracker.track(frame.armors, frame.t);

      if (!targets.empty()) {
        frame.target = targets.front();
        target_queue.push(targets.front());
        
        // 获取瞄准点（用于可视化）
        auto gs = gimbal.state();
        auto plan = planner.plan(std::make_optional(targets.front()), gs.bullet_speed);
        frame.aim_xyza = planner.debug_xyza;
      } else {
        frame.target = std::nullopt;
        target_queue.push(std::nullopt);
      }

      frame.track_end = std::chrono::steady_clock::now();
      
      // 发送到可视化线程
      visualization_queue.push(std::move(frame));
    }
    tools::logger()->info("[Track] Thread stopped");
  });

  // ========== 线程3: 云台控制线程 ==========
  auto plan_thread = std::thread([&]() {
    tools::logger()->info("[Plan] Thread started");
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, 
        plan.pitch, plan.pitch_vel, plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

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
        data["target_z"] = target->ekf_x()[4];
        data["target_vz"] = target->ekf_x()[5];
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data);

      // 保持100Hz控制频率
      std::this_thread::sleep_for(10ms);
    }
    tools::logger()->info("[Plan] Thread stopped");
  });

  // ========== 主线程: 可视化线程 ==========
  tools::logger()->info("[Main] Visualization thread running, press 'q' to exit");

  while (!exiter.exit()) {
    DebugFrame frame;
    if (visualization_queue.try_pop(frame)) {
      auto img = frame.img;
      auto gs = gimbal.state();

      // 绘制装甲板
      if (frame.target.has_value()) {
        auto target = frame.target.value();
        
        // 绘制所有装甲板（绿色）
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        // 绘制瞄准点（红色）
        auto image_points = solver.reproject_armor(
          frame.aim_xyza.head(3), frame.aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
        
        // 更新3D可视化
        auto plan = planner.plan(frame.target, gs.bullet_speed);
        Eigen::Vector3d target_xyz_in_world = frame.aim_xyza.head(3);
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
      }
      
      visualizer.show("3D Coordinate System");

      // 性能统计显示
      auto total_latency = tools::delta_time(frame.track_end, frame.camera_time);
      auto detect_time = tools::delta_time(frame.detect_end, frame.detect_start);
      auto track_time = tools::delta_time(frame.track_end, frame.detect_end);

      tools::draw_text(img, 
        fmt::format("Frame: {} | FPS: {:.1f}", frame.id, 1.0 / total_latency),
        {10, 30}, {255, 255, 0});
      tools::draw_text(img,
        fmt::format("Detect: {:.1f}ms | Track: {:.1f}ms | Total: {:.1f}ms", 
          detect_time * 1000, track_time * 1000, total_latency * 1000),
        {10, 60}, {255, 255, 0});
      tools::draw_text(img,
        fmt::format("Armors: {} | Targets: {}", 
          frame.armors.size(), frame.target.has_value() ? 1 : 0),
        {10, 90}, {255, 255, 0});

      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("Multi-threaded Debug", img);
      
      nlohmann::json perf_data;
      perf_data["fps"] = 1.0 / total_latency;
      perf_data["total_latency_ms"] = total_latency * 1000;
      perf_data["detect_time_ms"] = detect_time * 1000;
      perf_data["track_time_ms"] = track_time * 1000;
      plotter.plot(perf_data);
    }

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  // 优雅退出
  quit = true;
  
  if (capture_thread.joinable()) capture_thread.join();
  if (track_thread.joinable()) track_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  tools::logger()->info("Program exited successfully");

  return 0;
}
