#include "rclcpp/rclcpp.hpp"
#include "wust_binocular/camera/hik.hpp"
#include "wust_binocular/camera/video_player.hpp"
#include "wust_binocular/detect.hpp"
#include "wust_utils/ThreadPool.h"
class WustBinocularNode: public rclcpp::Node {
public:
    WustBinocularNode(const rclcpp::NodeOptions& options);
    ~WustBinocularNode();

private:
    void initcamera();
    void initSingleCamera(
        const std::string& prefix,
        bool& use_video,
        std::unique_ptr<HikCamera>& camera,
        std::unique_ptr<VideoPlayer>& video_player,
        cv::Mat& intrinsic,
        cv::Mat& distortion,
        double& alpha,
        int& beta
    );

    void frameCallback(ImageFrame& frame);
    void timerCallback();
    void stopTimer();
    void startTimer();
    void printStats();
    bool is_inited_ = false;
    std::unique_ptr<Detect> detect_;
    std::unique_ptr<HikCamera> camera_R_;
    std::unique_ptr<VideoPlayer> video_player_R_;
    std::unique_ptr<HikCamera> camera_L_;
    std::unique_ptr<VideoPlayer> video_player_L_;
    std::unique_ptr<ThreadPool> thread_pool_;
    cv::Mat camera_intrinsic_R_;
    cv::Mat camera_distortion_R_;
    bool use_video_R_;
    double video_alpha_R_;
    int video_beta_R_;
    cv::Mat camera_intrinsic_L_;
    cv::Mat camera_distortion_L_;
    bool use_video_L_;
    double video_alpha_L_;
    int video_beta_L_;
    rclcpp::TimerBase::SharedPtr timer_;
    int max_time_diff_us_;
    // 统计全部回调的帧数和时间点
    size_t passed_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;

    int fps;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };
    std::atomic<bool> timer_running_ { false };
    std::thread timer_thread_;
    int timer_count_ = 0;
    std::mutex timer_mtx_;
    std::condition_variable timer_cv_;
    double max_delay_;
};