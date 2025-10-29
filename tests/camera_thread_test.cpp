#include <fmt/core.h>
#include <unistd.h>

#include <chrono>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_pool.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/ascento.yaml | 位置参数，yaml配置文件路径 }";

tools::OrderedQueue frame_queue;

// 处理detect任务的线程函数
void detect_frame(tools::Frame && frame, auto_aim::YOLO & yolo)
{
  frame.armors = yolo.detect(frame.img);
  frame_queue.enqueue(std::move(frame));  // 使用移动语义
}

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  // tools::Recorder recorder(100);

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 处理线程函数
  auto process_thread = std::thread([&]() {
    tools::Frame process_frame;
    while (!exiter.exit()) {
      process_frame = frame_queue.dequeue();
      auto img = process_frame.img;
      auto armors = process_frame.armors;
      auto t = process_frame.t;

      nlohmann::json data;
      data["armor_num"] = armors.size();

      plotter.plot(data);
      // cv::resize(img, img, {}, 0.5, 0.5);
      // cv::imshow("reprojection", img);
    }
  });

  io::Camera camera(config_path);
  int num_yolo_thread = 8;
  
  // ✅ 使用新的YOLOThreadPool（推荐）
  tools::YOLOThreadPool yolo_pool(config_path, num_yolo_thread, true);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point last_t = std::chrono::steady_clock::now();

  int frame_id = 0;

  while (!exiter.exit()) {
    camera.read(img, t);
    auto dt = tools::delta_time(t, last_t);
    last_t = t;

    nlohmann::json data;
    data["fps"] = 1 / dt;

    frame_id++;

    // ✅ 使用移动语义，避免拷贝
    tools::Frame frame{frame_id, std::move(img), t};
    
    // ✅ 提交检测任务，自动管理YOLO实例
    yolo_pool.detect_async(std::move(frame), [](tools::Frame && processed_frame) {
      frame_queue.enqueue(std::move(processed_frame));
    });

    plotter.plot(data);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
