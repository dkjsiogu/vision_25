#include <fmt/core.h>
#include <fmt/format.h>

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/standard.yaml | 位置参数，yaml配置文件路径 }"
  "{num-threads    | 4                     | YOLO线程数 }";

// 检测帧结构
struct DetectionFrame
{
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  std::list<auto_aim::Armor> armors;
};

tools::ThreadSafeQueue<DetectionFrame, true> detection_queue(2);
tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);

int main(int argc, char * argv[])
{
  tools::Exiter exiter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  auto num_threads = cli.get<int>("num-threads");
  
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::logger()->info("Starting multi-threaded auto aim MPC with {} YOLO threads", num_threads);

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  // 创建YOLO线程池
  tools::YOLOThreadPool yolo_pool(config_path, num_threads, false);

  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;

  // ========== 线程1: 相机采集线程 ==========
  auto capture_thread = std::thread([&]() {
    tools::logger()->info("[Capture] Thread started");
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit) {
      camera.read(img, t);
      auto q = gimbal.q(t);
      
      tools::Frame yolo_frame;
      yolo_frame.img = std::move(img);
      yolo_frame.t = t;
      yolo_frame.q = q;

      // 提交到YOLO线程池
      yolo_pool.detect_async(std::move(yolo_frame), [t, q](tools::Frame && yolo_frame) {
        DetectionFrame processed_frame;
        processed_frame.img = std::move(yolo_frame.img);
        processed_frame.t = t;
        processed_frame.q = q;
        processed_frame.armors = std::move(yolo_frame.armors);
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
        target_queue.push(targets.front());
      } else {
        target_queue.push(std::nullopt);
      }
    }
    tools::logger()->info("[Track] Thread stopped");
  });

  // ========== 线程3: 云台控制线程 ==========
  auto plan_thread = std::thread([&]() {
    tools::logger()->info("[Plan] Thread started");

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      // [改进] 使用云台实测状态作为 MPC 初值
      auto plan = planner.plan(target, gs.bullet_speed, gs);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, 
        plan.pitch, plan.pitch_vel, plan.pitch_acc);

      // 保持100Hz控制频率
      std::this_thread::sleep_for(10ms);
    }
    tools::logger()->info("[Plan] Thread stopped");
  });

  // ========== 主线程: 监控和退出 ==========
  tools::logger()->info("[Main] Press Ctrl+C to exit");
  
  while (!exiter.exit()) {
    std::this_thread::sleep_for(100ms);
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
