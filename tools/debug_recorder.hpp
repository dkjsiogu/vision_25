#ifndef TOOLS__DEBUG_RECORDER_HPP
#define TOOLS__DEBUG_RECORDER_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace tools
{

/**
 * 通用调试数据记录器
 *
 * 用法：
 *   auto& rec = tools::DebugRecorder::instance("outpost");
 *   rec.enable("outpost_log.csv");
 *
 *   // 每帧记录
 *   rec.set("cx", 3.5);
 *   rec.set("cy", 1.2);
 *   rec.set("valid", true);
 *   rec.commit();  // 写入一行
 */
class DebugRecorder
{
public:
  DebugRecorder() = default;

  // 获取命名实例（支持多个独立的记录器）
  static DebugRecorder & instance(const std::string & name = "default")
  {
    static std::map<std::string, DebugRecorder> instances;
    return instances[name];
  }

  // 启用记录，指定输出文件
  void enable(const std::string & path)
  {
    if (file_.is_open()) file_.close();
    file_.open(path);
    enabled_ = file_.is_open();
    header_written_ = false;
    start_time_ = std::chrono::steady_clock::now();
    frame_ = 0;
    if (enabled_) {
      // 预设时间字段
      field_order_.clear();
      field_order_.push_back("frame");
      field_order_.push_back("t_ms");
    }
  }

  void disable()
  {
    if (file_.is_open()) file_.close();
    enabled_ = false;
  }

  bool is_enabled() const { return enabled_; }

  // 设置字段值（支持多种类型）
  void set(const std::string & key, double value)
  {
    if (!enabled_) return;
    add_field(key);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << value;
    current_row_[key] = ss.str();
  }

  void set(const std::string & key, int value)
  {
    if (!enabled_) return;
    add_field(key);
    current_row_[key] = std::to_string(value);
  }

  void set(const std::string & key, bool value)
  {
    if (!enabled_) return;
    add_field(key);
    current_row_[key] = value ? "1" : "0";
  }

  void set(const std::string & key, const std::string & value)
  {
    if (!enabled_) return;
    add_field(key);
    current_row_[key] = value;
  }

  // 提交当前行（写入文件）
  void commit()
  {
    if (!enabled_) return;

    frame_++;
    auto now = std::chrono::steady_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(now - start_time_).count();

    current_row_["frame"] = std::to_string(frame_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << t_ms;
    current_row_["t_ms"] = ss.str();

    // 首次提交时写入表头
    if (!header_written_) {
      for (size_t i = 0; i < field_order_.size(); i++) {
        if (i > 0) file_ << ",";
        file_ << field_order_[i];
      }
      file_ << "\n";
      header_written_ = true;
    }

    // 写入数据行
    for (size_t i = 0; i < field_order_.size(); i++) {
      if (i > 0) file_ << ",";
      auto it = current_row_.find(field_order_[i]);
      if (it != current_row_.end()) {
        file_ << it->second;
      }
    }
    file_ << "\n";
    file_.flush();

    current_row_.clear();
  }

private:
  void add_field(const std::string & key)
  {
    if (field_set_.find(key) == field_set_.end()) {
      field_set_.insert(key);
      field_order_.push_back(key);
    }
  }

  bool enabled_ = false;
  bool header_written_ = false;
  std::ofstream file_;
  std::chrono::steady_clock::time_point start_time_;
  int frame_ = 0;

  std::vector<std::string> field_order_;
  std::set<std::string> field_set_;
  std::map<std::string, std::string> current_row_;
};

}  // namespace tools

#endif  // TOOLS__DEBUG_RECORDER_HPP
