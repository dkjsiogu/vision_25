#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "outpost_target.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

  // 获取前哨站追踪状态 (用于判断是否允许开火)
  bool is_outpost_tracking() const { return is_tracking_outpost_ && outpost_target_.is_tracking(); }
  const OutpostTarget & outpost_target() const { return outpost_target_; }

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  // 前哨站专用追踪器
  OutpostTarget outpost_target_;
  bool is_tracking_outpost_ = false;
  std::string config_path_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  // 前哨站专用处理
  bool handle_outpost(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
  Target outpost_to_target(std::chrono::steady_clock::time_point t) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP