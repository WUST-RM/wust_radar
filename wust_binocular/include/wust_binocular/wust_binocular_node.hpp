#include "rclcpp/rclcpp.hpp"
#include "wust_binocular/camera/hik.hpp"
#include "wust_binocular/camera/video_player.hpp"
#include "wust_binocular/detect.hpp"
#include "wust_binocular/frame_synchronizer.hpp"
#include "wust_binocular/stereo_depth_estimator.hpp"
#include "wust_utils/ThreadPool.h"
#include <wust_utils/logger.hpp>
#include <wust_utils/timer.hpp>
#include <yaml-cpp/yaml.h>
using FrameCallback = std::function<void(ImageFrame&, bool)>;
class SingleCamera {
public:
    SingleCamera(
        const YAML::Node& camera_config,
        const std::string key,
        FrameCallback cb,
        bool use_trigger
    ) {
        this->callback_ = cb;
        use_trigger_ = use_trigger;
        use_video_ = camera_config[key]["video_player"]["use"].as<bool>(false);
        if (use_video_) {
            std::string video_play_path =
                camera_config[key]["video_player"]["path"].as<std::string>("");
            int video_play_fps = camera_config[key]["video_player"]["fps"].as<int>(30);
            int start_frame = camera_config[key]["video_player"]["start_frame"].as<int>(0);
            bool loop = camera_config[key]["video_player"]["loop"].as<bool>(false);
            video_player_ =
                std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
            video_player_->enablehighPriorityAndCpuidPriority(
                camera_config[key]["video_player"]["use_high_priority"].as<bool>(false),
                camera_config[key]["video_player"]["high_priority_cpu_id"].as<int>(0),
                camera_config[key]["video_player"]["high_priority_cpu_priority"].as<int>(0),
                camera_config[key]["video_player"]["use_sched_fifo"].as<bool>(false)
            );
            video_player_->setCallback([this](ImageFrame& frame) { callback_(frame, use_video_); });
            if (use_trigger_) {
                video_player_->enableTriggerMode(true);
            }

        } else {
            camera_ = std::make_unique<HikCamera>();
            std::string target_sn = camera_config[key]["target_sn"].as<std::string>();
            if (!camera_->initializeCamera(target_sn)) {
                WUST_ERROR("single_camera") << "Camera initialization failed.";
                return;
            }

            camera_->setParameters(
                camera_config[key]["acquisition_frame_rate"].as<int>(),
                camera_config[key]["exposure_time"].as<int>(),
                camera_config[key]["gain"].as<double>(),
                camera_config[key]["gamma"].as<double>(),
                camera_config[key]["adc_bit_depth"].as<std::string>(),
                camera_config[key]["pixel_format"].as<std::string>(),
                camera_config[key]["acquisition_frame_rate_enable"].as<bool>(),
                camera_config[key]["reverse_x"].as<bool>(false),
                camera_config[key]["reverse_y"].as<bool>(false)
            );
            camera_->enablehighPriorityAndCpuidPriority(
                camera_config[key]["use_high_priority"].as<bool>(false),
                camera_config[key]["high_priority_cpu_id"].as<int>(0),
                camera_config[key]["high_priority_cpu_priority"].as<int>(0),
                camera_config[key]["use_sched_fifo"].as<bool>(false)
            );
            camera_->setFrameCallback([this](ImageFrame& frame) { callback_(frame, use_video_); });
            if (use_trigger_) {
                camera_->enableTrigger(TriggerType::Software, "Software", 0);
            }
            if_recorder_ = camera_config[key]["recorder"].as<bool>(false);
        }
        const std::string camera_info_path =
            camera_config[key]["camera_info_path"].as<std::string>();
        YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
        std::vector<double> camera_k =
            config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        camera_intrinsic_ = K.clone();
        camera_distortion_ = D.clone();
    };
    ~SingleCamera() {
        if (use_trigger_) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cv_.notify_one();
            thread_.join();
        }
        if (use_video_) {
            video_player_->stop();
        } else {
            if (camera_ && !use_trigger_) {
                camera_->stopCamera();
            }
            camera_.reset();
        }
    }
    void start() {
        if (video_player_ && use_video_) {
            video_player_->start();
        }
        if (camera_ && !use_video_) {
            camera_->startCamera(if_recorder_);
        }
        if (use_trigger_) {
            thread_ = std::thread([this]() { this->loop(); });
        }
    }
    void trigger() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            triggered_ = true;
        }
        cv_.notify_one();
    }
    FrameCallback callback_;

public:
    std::unique_ptr<HikCamera> camera_;
    std::unique_ptr<VideoPlayer> video_player_;
    cv::Mat camera_intrinsic_;
    cv::Mat camera_distortion_;
    bool use_video_;
    bool start_ = false;
    bool if_recorder_;
    bool use_trigger_;

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool triggered_;
    bool stop_ = false;
    void loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() { return triggered_ || stop_; });

            if (stop_)
                break;

            triggered_ = false;
            lock.unlock();
            doWork();
        }
    }

    void doWork() {
        if (camera_) {
            camera_->read();
        }
        if (video_player_) {
            video_player_->read();
        }
    }
};
class WustBinocularNode: public rclcpp::Node {
public:
    WustBinocularNode(const rclcpp::NodeOptions& options);
    ~WustBinocularNode();

private:
    void initcamera();
    void initLog();
    void loadCommonParams();
    void frameCallback(const CommonFrame& frame);
    void timerCallback(double dt_ms);
    void detectCallback(
        const CommonFrame& frame,
        const std::vector<Car>& cars,
        DetectDebug& detect_debug
    );
    void printStats();
    bool is_inited_ = false;
    bool use_binocular_ = false;
    bool use_id_match_ = false;
    bool use_trigger_ = false;

    std::unique_ptr<Detect> detect_;
    std::unique_ptr<SingleCamera> camera_R_;
    std::unique_ptr<SingleCamera> camera_L_;
    std::unique_ptr<ThreadPool> thread_pool_;
    double max_time_diff_ms_;
    size_t passed_count_ = 0;
    std::chrono::steady_clock::time_point last_stat_time_steady_;
    int fps_;
    std::mutex callback_mutex_;
    std::atomic<int> infer_running_count_ { 0 };
    // std::atomic<bool> timer_running_ { false };
    // std::thread timer_thread_;
    std::unique_ptr<Timer> timer_;
    int timer_count_ = 0;
    // std::mutex timer_mtx_;
    // std::condition_variable timer_cv_;
    double max_delay_;
    uint64_t seq_id_R_ = 0;
    uint64_t seq_id_L_ = 0;
    bool seq_reset_R_ = false;
    bool seq_reset_L_ = false;
    std::unique_ptr<FrameSynchronizer> synchronizer_;
    std::unique_ptr<StereoDepthEstimator> depth_estimator_;
};