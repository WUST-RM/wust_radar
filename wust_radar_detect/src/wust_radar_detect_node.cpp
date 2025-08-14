#include "wust_radar_detect/wust_radar_detect_node.hpp"
#include "wust_radar_detect/type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <wust_utils/logger.hpp>
WustRadarDetectNode::WustRadarDetectNode(const rclcpp::NodeOptions& options):
    Node("wust_radar_detect_node", options) {
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
        &WustRadarDetectNode::detectCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency() * 2);
    initcamera();
    if (use_trigger_) {
        timer_ = std::make_unique<Timer>();
    }
    if (camera_)
        camera_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    result_pub_ = this->create_publisher<wust_radar_interfaces::msg::DetectResult>(
        "detect_result",
        rclcpp::SensorDataQoS()
    );
    is_inited_ = true;
    if (timer_) {
        auto timercallback =
            std::bind(&WustRadarDetectNode::timerCallback, this, std::placeholders::_1);
        double rate_hz = static_cast<double>(fps_);
        timer_->start(rate_hz, timercallback);
    }
}
void WustRadarDetectNode::loadCommonParams() {
    this->declare_parameter<double>("common.max_delay", 1.0);
    this->declare_parameter<int>("common.fps", 60);
    this->declare_parameter<bool>("common.use_trigger", false);
    this->declare_parameter<bool>("common.use_binocular", false);
    this->declare_parameter<bool>("common.use_id_match", false);

    this->get_parameter("common.max_delay", max_delay_);
    this->get_parameter("common.fps", fps_);
    this->get_parameter("common.use_trigger", use_trigger_);
    this->get_parameter("common.use_binocular", use_binocular_);
    this->get_parameter("common.use_id_match", use_id_match_);
}
void WustRadarDetectNode::initLog() {
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
WustRadarDetectNode::~WustRadarDetectNode() {
    if (timer_) {
        timer_->stop();
    }
}
void WustRadarDetectNode::frameCallback(const ImageFrame& f, bool use_video) {
    timer_count_++;
    printStats();
    if (detect_ && detect_->resource_pool_->idleCount() == 0) {
        return;
    }

    cv::Mat img;
    if (use_video) {
        //img = convertToMatrgb(f);
        if (!f.src_img.empty()) {
            img = std::move(f.src_img);
        }

    } else {
        img = convertToMat(f);
    }

    CommonFrame frame;
    frame.image = std::move(img);
    frame.timestamp = f.timestamp;
    passed_count_++;
    // auto now = std::chrono::steady_clock::now();
    // std::cout<<time_utils::durationMs(f.timestamp, now)<<std::endl;
    if (detect_) {
        detect_->pushInput(frame);
    }
}
static wust_radar_interfaces::msg::Box yoloBox2msgBox(const yolo::Box& box) {
    auto msg_box = wust_radar_interfaces::msg::Box();
    msg_box.top = box.top;
    msg_box.bottom = box.bottom;
    msg_box.left = box.left;
    msg_box.right = box.right;
    return msg_box;
}

void WustRadarDetectNode::detectCallback(
    const CommonFrame& frame,
    const Cars& cars,
    DetectDebug& detect_debug
) {
    auto msg = wust_radar_interfaces::msg::DetectResult();
    for (auto car: cars.cars) {
        auto msg_car = wust_radar_interfaces::msg::SingleCar();
        msg_car.car_box = yoloBox2msgBox(car.car);
        msg_car.confidence = car.car.confidence;
        for (auto armor: car.armors) {
            auto msg_armor = wust_radar_interfaces::msg::Armor();
            msg_armor.box = yoloBox2msgBox(armor);
            msg_armor.number = armor.class_label;
            msg_armor.color = armor.color;
            msg_armor.confidence = armor.confidence;
            msg_car.armors.push_back(msg_armor);
        }
        msg.cars.push_back(msg_car);
    }
    auto now = std::chrono::steady_clock::now();
    double delay_ms = time_utils::durationMs(frame.timestamp, now);
    rclcpp::Duration delay_ros =
        rclcpp::Duration::from_nanoseconds(static_cast<int64_t>(delay_ms * 1'000'000));
    msg.header.stamp = this->now() - delay_ros;
    msg.header.frame_id = "camera";
    result_pub_->publish(msg);
    showDebug(detect_debug);
}
void WustRadarDetectNode::timerCallback(double dt_ms) {
    if (camera_)
        camera_->trigger();
}

void WustRadarDetectNode::printStats() {
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
                    std::bind(&WustRadarDetectNode::timerCallback, this, std::placeholders::_1);
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
void WustRadarDetectNode::initcamera() {
    this->declare_parameter<std::string>("camera_config_file", "");
    std::string camera_config_file = this->get_parameter("camera_config_file").as_string();
    auto camera_config = YAML::LoadFile(camera_config_file);
    FrameCallback cb_ = [this](const ImageFrame& f, bool use_video) {
        if (!is_inited_) {
            return;
        }

        thread_pool_->enqueue(
            [frame = std::move(f), use_video, this]() { this->frameCallback(frame, use_video); },
            -1
        );
    };

    camera_ = std::make_unique<SingleCamera>(camera_config, "camera", cb_, use_trigger_);
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(WustRadarDetectNode)