#include "rclcpp/rclcpp.hpp"
#include "wust_radar_detect/camera/single_camera.hpp"
#include "wust_radar_detect/detect.hpp"
#include "wust_radar_interfaces/msg/detect_result.hpp"
#include "wust_utils/ThreadPool.h"
#include <wust_utils/logger.hpp>
#include <wust_utils/timer.hpp>
#include <yaml-cpp/yaml.h>

class WustRadarDetectNode: public rclcpp::Node {
public:
    WustRadarDetectNode(const rclcpp::NodeOptions& options);
    ~WustRadarDetectNode();

private:
    void initcamera();
    void initLog();
    void loadCommonParams();
    void frameCallback(const ImageFrame& f, bool use_video);
    void timerCallback(double dt_ms);
    void detectCallback(
        const CommonFrame& frame,
        const Cars& cars,
        DetectDebug& detect_debug
    );
    void printStats();
    bool is_inited_ = false;
    bool use_binocular_ = false;
    bool use_id_match_ = false;
    bool use_trigger_ = false;

    rclcpp::Publisher<wust_radar_interfaces::msg::DetectResult>::SharedPtr result_pub_;
    std::unique_ptr<Detect> detect_;
    std::unique_ptr<SingleCamera> camera_;
    std::unique_ptr<ThreadPool> thread_pool_;
    size_t passed_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    int fps_;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };
    std::unique_ptr<Timer> timer_;
    int timer_count_ = 0;
    double max_delay_;
};