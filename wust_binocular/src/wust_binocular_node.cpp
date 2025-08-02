#include "wust_binocular/wust_binocular_node.hpp"
#include "wust_binocular/type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <wust_utils/logger.hpp>
WustBinocularNode::WustBinocularNode(const rclcpp::NodeOptions& options):
    Node("wust_binocular_node", options) {
    initLog();
    loadCommonParams();

    DetectConfig detect_config;
    this->declare_parameter<std::string>("detect.config_path", "");
    this->get_parameter("detect.config_path", detect_config.config_path);
    this->declare_parameter<int>("detect.max_infer_threads", 4);
    this->get_parameter("detect.max_infer_threads", detect_config.max_infer_threads);
    this->declare_parameter<double>("detect.min_free_mem_ratio", 0.5);
    this->get_parameter("detect.min_free_mem_ratio", detect_config.min_free_mem_ratio);
    detect_ = std::make_unique<Detect>(detect_config);
    detect_->setCallback(std::bind(
        &WustBinocularNode::detectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency() * 2);
    initcamera();
    if (use_binocular_) {
        double max_time_diff_s_ = max_time_diff_ms_ * 1000;
        synchronizer_ = std::make_unique<FrameSynchronizer>(max_time_diff_s_, use_id_match_);
        MatchFrameCallback match_frame_callback =
            [this](const FrameUnmatched& left, const FrameUnmatched& right) {
                CommonFrame frame;
                frame.image_R = right.src_img;
                frame.image_L = left.src_img;
                frame.timestamp_R = right.timestamp;
                frame.timestamp_L = left.timestamp;
                this->frameCallback(frame);
            };
        synchronizer_->setMatchCallback(match_frame_callback);
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // 单位矩阵，表示无旋转
        cv::Mat T = (cv::Mat_<double>(3, 1) << 0.06, 0.0, 0.0); // 6cm 平移
        depth_estimator_ = std::make_unique<StereoDepthEstimator>(
            camera_L_->camera_intrinsic_,
            camera_L_->camera_distortion_,
            camera_R_->camera_intrinsic_,
            camera_R_->camera_distortion_,
            R,
            T
        );
    }
    if (use_trigger_) {
        //startTimer();
        timer_ = std::make_unique<Timer>();
    }
    if (camera_L_)
        camera_L_->start();
    if (camera_R_)
        camera_R_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    seq_reset_R_ = true;
    seq_reset_L_ = true;
    is_inited_ = true;
    if (timer_) {
        auto timercallback =
            std::bind(&WustBinocularNode::timerCallback, this, std::placeholders::_1);
        double rate_hz = static_cast<double>(fps_);
        timer_->start(rate_hz, timercallback);
    }
}
void WustBinocularNode::loadCommonParams() {
    this->declare_parameter<double>("common.max_delay", 1.0);
    this->declare_parameter<double>("common.max_time_diff_ms", 10.0);
    this->declare_parameter<int>("common.fps", 60);
    this->declare_parameter<bool>("common.use_trigger", false);
    this->declare_parameter<bool>("common.use_binocular", false);
    this->declare_parameter<bool>("common.use_id_match", false);

    this->get_parameter("common.max_delay", max_delay_);
    this->get_parameter("common.max_time_diff_ms", max_time_diff_ms_);
    this->get_parameter("common.fps", fps_);
    this->get_parameter("common.use_trigger", use_trigger_);
    this->get_parameter("common.use_binocular", use_binocular_);
    this->get_parameter("common.use_id_match", use_id_match_);
}
void WustBinocularNode::initLog() {
    this->declare_parameter<std::string>("logger.log_level", "INFO");
    this->declare_parameter<std::string>("logger.log_path", "wust_log");
    this->declare_parameter<bool>("logger.use_logcli", true);
    this->declare_parameter<bool>("logger.use_logfile", false);
    this->declare_parameter<bool>("logger.use_simplelog", false);

    std::string log_level = this->get_parameter("logger.log_level").as_string();
    std::string log_path = this->get_parameter("logger.log_path").as_string();
    bool use_logcli = this->get_parameter("logger.use_logcli").as_bool();
    bool use_logfile = this->get_parameter("logger.use_logfile").as_bool();
    bool use_simplelog = this->get_parameter("logger.use_simplelog").as_bool();
    initLogger(log_level, log_path, use_logcli, use_logfile, use_simplelog);
}
WustBinocularNode::~WustBinocularNode() {
    if (timer_) {
        timer_->stop();
    }
}
void WustBinocularNode::frameCallback(const CommonFrame& frame) {
    timer_count_++;
    printStats();
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - frame.timestamp_L).count();
    if (dt > max_delay_) {
        return;
    }

    if (use_binocular_) {
        double frame_dt_ms =
            std::abs(std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                         frame.timestamp_L - frame.timestamp_R
            )
                         .count());

        if (frame_dt_ms > max_time_diff_ms_) {
            return;
        }
        WUST_DEBUG("frameCallback") << "frame_dt: " << frame_dt_ms << " ms";
    }

    passed_count_++;
    if (detect_) {
        detect_->pushInput(frame);
    }
}

void WustBinocularNode::detectCallback(
    const CommonFrame& frame,
    const std::vector<Car>& cars,
    DetectDebug& detect_debug
) {
    if (depth_estimator_) {
    }
    showDebug(detect_debug);
}
void WustBinocularNode::timerCallback(double dt_ms) {
    if (camera_L_)
        camera_L_->trigger();
    if (camera_R_)
        camera_R_->trigger();
}

void WustBinocularNode::printStats() {
    static int timer_check_count = 0;
    using namespace std::chrono;

    auto now = steady_clock::now();

    if (last_stat_time_steady_.time_since_epoch().count() == 0) {
        last_stat_time_steady_ = now;
        return;
    }

    auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
    if (elapsed.count() >= 1.0) {
        if (timer_count_ < fps_ / 10) {
            timer_check_count++;
        }
        if (timer_check_count > 5 && use_trigger_) {
            timer_check_count = 0;
            if (timer_) {
                auto timercallback =
                    std::bind(&WustBinocularNode::timerCallback, this, std::placeholders::_1);
                double rate_hz = static_cast<double>(fps_);
                timer_->start(rate_hz, timercallback);
            }
        }
        WUST_INFO("printStats") << "tc: " << timer_count_ << " ,pass: " << passed_count_
                                << " , det: " << detect_->detect_finish_count_;
        timer_count_ = 0;
        passed_count_ = 0;
        detect_->detect_finish_count_ = 0;
        last_stat_time_steady_ = now;
    }
}
void WustBinocularNode::initcamera() {
    this->declare_parameter<std::string>("camera_config_file", "");
    std::string camera_config_file = this->get_parameter("camera_config_file").as_string();
    auto camera_config = YAML::LoadFile(camera_config_file);
    FrameCallback cb_R = [this](const ImageFrame& f, bool use_video) {
        if (!is_inited_ || !use_binocular_) {
            return;
        }
        if (seq_reset_R_) {
            seq_id_R_ = 0;
            seq_reset_R_ = false;
        }
        cv::Mat img;
        if (use_video) {
            img = convertToMatrgb(f);
        } else {
            img = convertToMatbgr(f);
        }
        FrameUnmatched frame_unmatched = { .seq_id = seq_id_R_++,
                                           .src_img = img,
                                           .timestamp = f.timestamp,
                                           .source = FrameSource::RIGHT };
        if (synchronizer_) {
            synchronizer_->pushFrame(frame_unmatched);
        }
    };
    FrameCallback cb_L = [this](const ImageFrame& f, bool use_video) {
        if (!is_inited_) {
            return;
        }
        cv::Mat img;
        if (use_video) {
            img = convertToMatrgb(f);
        } else {
            img = convertToMatbgr(f);
        }
        if (use_binocular_) {
            if (seq_reset_L_) {
                seq_id_L_ = 0;
                seq_reset_L_ = false;
            }
            FrameUnmatched frame_unmatched = { .seq_id = seq_id_L_++,
                                               .src_img = img,
                                               .timestamp = f.timestamp,
                                               .source = FrameSource::LEFT };
            if (synchronizer_) {
                synchronizer_->pushFrame(frame_unmatched);
            }
        } else {
            CommonFrame frame;
            frame.image_L = img;
            frame.timestamp_L = f.timestamp;
            // thread_pool_->enqueue(
            //             [frame = std::move(frame), this]() {
            this->frameCallback(frame);
            //      },
            //     -1
            // );
        }
    };
    if (use_binocular_) {
        camera_R_ = std::make_unique<SingleCamera>(camera_config, "cameraR", cb_R, use_trigger_);
    }

    camera_L_ = std::make_unique<SingleCamera>(camera_config, "cameraL", cb_L, use_trigger_);
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(WustBinocularNode)