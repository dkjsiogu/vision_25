#ifndef TOOLS__THREAD_POOL_HPP
#define TOOLS__THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"

namespace tools
{
struct Frame
{
  int id;
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  std::list<auto_aim::Armor> armors;
};

inline std::vector<auto_aim::YOLO> create_yolo11s(
  const std::string & config_path, int number, bool debug)
{
  std::vector<auto_aim::YOLO> yolo11s;
  yolo11s.reserve(number);
  for (int i = 0; i < number; i++) {
    yolo11s.push_back(auto_aim::YOLO(config_path, debug));
  }
  return yolo11s;
}

inline std::vector<auto_aim::YOLO> create_yolov8s(
  const std::string & config_path, int number, bool debug)
{
  std::vector<auto_aim::YOLO> yolov8s;
  yolov8s.reserve(number);
  for (int i = 0; i < number; i++) {
    yolov8s.push_back(auto_aim::YOLO(config_path, debug));
  }
  return yolov8s;
}

class OrderedQueue
{
public:
  OrderedQueue() : current_id_(1) {}
  ~OrderedQueue()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      main_queue_ = std::queue<tools::Frame>();
      buffer_.clear();
      current_id_ = 0;
    }
    tools::logger()->info("OrderedQueue destroyed, queue and buffer cleared.");
  }

  // 使用移动语义避免拷贝
  void enqueue(tools::Frame && item)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (item.id < current_id_) {
      tools::logger()->warn("Frame id {} < current_id {}, dropped", item.id, current_id_);
      return;
    }

    if (item.id == current_id_) {
      main_queue_.push(std::move(item));
      current_id_++;

      // 处理buffer中连续的帧
      auto it = buffer_.find(current_id_);
      while (it != buffer_.end()) {
        main_queue_.push(std::move(it->second));
        buffer_.erase(it);
        current_id_++;
        it = buffer_.find(current_id_);
      }

      cond_var_.notify_one();
    } else {
      // 乱序到达，暂存到buffer
      buffer_[item.id] = std::move(item);
    }
  }

  // 阻塞获取
  tools::Frame dequeue()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this]() { return !main_queue_.empty(); });
    
    tools::Frame item = std::move(main_queue_.front());
    main_queue_.pop();
    return item;
  }

  // 非阻塞获取
  bool try_dequeue(tools::Frame & item)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (main_queue_.empty()) {
      return false;
    }
    item = std::move(main_queue_.front());
    main_queue_.pop();
    return true;
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_queue_.size() + buffer_.size();
  }

private:
  std::queue<tools::Frame> main_queue_;
  std::unordered_map<int, tools::Frame> buffer_;
  int current_id_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
};

class ThreadPool
{
public:
  explicit ThreadPool(size_t num_threads) : stop_(false)
  {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
              return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    
    for (std::thread & worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  // 禁用拷贝和移动
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool & operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool & operator=(ThreadPool &&) = delete;

  // 提交任务并返回future（可获取返回值）
  template <class F, class... Args>
  auto enqueue(F && f, Args &&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
  {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }

      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    
    return res;
  }

  // 获取线程池大小
  size_t size() const { return workers_.size(); }

  // 获取待处理任务数
  size_t pending_tasks() const
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size();
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  mutable std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

// YOLO专用线程池（对象池模式）
class YOLOThreadPool
{
public:
  YOLOThreadPool(const std::string & config_path, size_t num_workers, bool debug = false)
  : stop_(false)
  {
    // 创建YOLO实例池
    yolos_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++) {
      yolos_.emplace_back(config_path, debug);
      available_yolos_.push(i);
    }

    // 创建工作线程
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
              return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~YOLOThreadPool()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    
    for (std::thread & worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  // 提交检测任务
  void detect_async(tools::Frame && frame, std::function<void(tools::Frame &&)> callback)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      
      if (stop_) {
        throw std::runtime_error("enqueue on stopped YOLOThreadPool");
      }

      tasks_.emplace([this, frame = std::move(frame), callback = std::move(callback)]() mutable {
        // 获取可用的YOLO实例
        size_t yolo_id;
        {
          std::unique_lock<std::mutex> lock(yolo_mutex_);
          yolo_cv_.wait(lock, [this] { return !available_yolos_.empty(); });
          yolo_id = available_yolos_.front();
          available_yolos_.pop();
        }

        // 执行检测
        frame.armors = yolos_[yolo_id].detect(frame.img);

        // 释放YOLO实例
        {
          std::lock_guard<std::mutex> lock(yolo_mutex_);
          available_yolos_.push(yolo_id);
        }
        yolo_cv_.notify_one();

        // 执行回调
        callback(std::move(frame));
      });
    }
    condition_.notify_one();
  }

  size_t size() const { return workers_.size(); }

private:
  std::vector<auto_aim::YOLO> yolos_;
  std::queue<size_t> available_yolos_;
  std::mutex yolo_mutex_;
  std::condition_variable yolo_cv_;

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

}  // namespace tools

#endif  // TOOLS__THREAD_POOL_HPP
