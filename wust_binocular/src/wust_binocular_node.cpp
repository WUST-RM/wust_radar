#include "wust_binocular/wust_binocular_node.hpp"
#include "wust_binocular/type/type.hpp"
#include "yaml-cpp/yaml.h"
#include <wust_utils/logger.hpp>
WustBinocularNode::WustBinocularNode(const rclcpp::NodeOptions& options):
    Node("wust_binocular_node", options) {
    // 声明参数（带默认值）
    this->declare_parameter<std::string>("logger.log_level", "INFO");
    this->declare_parameter<std::string>("logger.log_path", "wust_log");
    this->declare_parameter<bool>("logger.use_logcli", true);
    this->declare_parameter<bool>("logger.use_logfile", false);
    this->declare_parameter<bool>("logger.use_simplelog", false);

    // 获取参数
    std::string log_level = this->get_parameter("logger.log_level").as_string();
    std::string log_path = this->get_parameter("logger.log_path").as_string();
    bool use_logcli = this->get_parameter("logger.use_logcli").as_bool();
    bool use_logfile = this->get_parameter("logger.use_logfile").as_bool();
    bool use_simplelog = this->get_parameter("logger.use_simplelog").as_bool();

    // 初始化日志系统
    initLogger(log_level, log_path, use_logcli, use_logfile, use_simplelog);
    DetectConfig detect_config;
    this->declare_parameter<int>("common.max_infer_threads", 4);
    this->get_parameter("common.max_infer_threads", detect_config.max_infer_threads);
    detect_ = std::make_unique<Detect>(detect_config);
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency() * 2);

    this->declare_parameter<double>("common.max_delay", 1.0);
    this->get_parameter("common.max_delay", this->max_delay_);

    initcamera();
    this->declare_parameter<int>("common.max_time_diff_us", 100);
    this->get_parameter("common.max_time_diff_us", this->max_time_diff_us_);
    this->declare_parameter<int>("common.fps", 60);
    this->get_parameter("common.fps", fps);
    if (video_player_R_ && use_video_R_) {
        video_player_R_->start();
    }
    if (camera_R_ && !use_video_R_) {
        bool if_recorder;
        this->declare_parameter<bool>("cameraR.recorder", false);
        this->get_parameter("cameraR.recorder", if_recorder);
        camera_R_->startCamera(if_recorder);
    }
    if (video_player_L_ && use_video_L_) {
        video_player_L_->start();
    }
    if (camera_L_ && !use_video_L_) {
        bool if_recorder;
        this->declare_parameter<bool>("cameraL.recorder", false);
        this->get_parameter("cameraL.recorder", if_recorder);
        camera_L_->startCamera(if_recorder);
    }
    // timer_ = this->create_wall_timer(
    //     std::chrono::milliseconds(1000 / fps),
    //     std::bind(&WustBinocularNode::timerCallback, this)
    // );
    //startTimer();
    is_inited_ = true;
}
WustBinocularNode::~WustBinocularNode() {
    //stopTimer();
    if (use_video_R_) {
        video_player_R_->stop();
    } else {
        if (camera_R_) {
            camera_R_->stopCamera();
            camera_R_.reset();
        }
    }
    if (use_video_L_) {
        video_player_L_->stop();
    } else {
        if (camera_L_) {
            camera_L_->stopCamera();
            camera_L_.reset();
        }
    }
}
void WustBinocularNode::frameCallback(ImageFrame& frame_L) {
    auto start = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(start - frame_L.timestamp).count();
    timer_count_++;
    if (dt > max_delay_) {
        return;
    }

    printStats();
    cv::Mat img_L;
    if (!use_video_L_) {
        img_L = convertToMatrgb(frame_L);
    } else {
        img_L = convertToMatrgb(frame_L);
        img_L.convertTo(img_L, -1, video_alpha_L_, video_beta_L_);
    }
    passed_count_++;
    if (!img_L.empty()) {
        CommonFrame commonframe;
        commonframe.image_L = img_L.clone();
        //commonframe.image_R = img_R.clone();
        commonframe.timestamp_L = frame_L.timestamp;
        // commonframe.timestamp_R = frame_R.timestamp;
        if (detect_) {
            //thread_pool_->enqueue([&commonframe, this]() { detect_->pushInput(commonframe); }, -1);
            detect_->pushInput(commonframe);
        }
        // cv::imshow("left", img_L);
        // cv::imshow("right", img_R);
        // cv::waitKey(1);
    }
    auto end = std::chrono::steady_clock::now();
    //std::cout<<"time:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<std::endl;
}
void WustBinocularNode::timerCallback() {
    using namespace std::chrono;

    timer_count_++;
    printStats();
    ImageFrame frame_R;
    ImageFrame frame_L;
    std::thread tR([&]() {
        if (use_video_R_ && video_player_R_) {
            frame_R = video_player_R_->readImage();
        } else if (camera_R_) {
            frame_R = camera_R_->readImage();
        }
    });
    std::thread tL([&]() {
        if (use_video_L_ && video_player_L_) {
            frame_L = video_player_L_->readImage();
        } else if (camera_L_) {
            frame_L = camera_L_->readImage();
        }
    });
    tR.join();
    tL.join();

    auto epoch = steady_clock::time_point {};
    auto ts_L_us = duration_cast<microseconds>(frame_L.timestamp - epoch).count();
    auto ts_R_us = duration_cast<microseconds>(frame_R.timestamp - epoch).count();
    int dt = std::abs(ts_R_us - ts_L_us);

    if (dt >= max_time_diff_us_) {
        return;
    }

    // 统计通过时间差筛选的帧数和计算有效帧率
    passed_count_++;

    cv::Mat img_R;
    if (!use_video_R_) {
        img_R = convertToMatrgb(frame_R);
    } else {
        img_R = convertToMatrgb(frame_R);
        img_R.convertTo(img_R, -1, video_alpha_R_, video_beta_R_);
    }
    cv::Mat img_L;
    if (!use_video_L_) {
        img_L = convertToMatrgb(frame_L);
    } else {
        img_L = convertToMatrgb(frame_L);
        img_L.convertTo(img_L, -1, video_alpha_L_, video_beta_L_);
    }

    if (!img_L.empty() && !img_R.empty()) {
        CommonFrame commonframe;
        commonframe.image_L = img_L.clone();
        commonframe.image_R = img_R.clone();
        commonframe.timestamp_L = frame_L.timestamp;
        commonframe.timestamp_R = frame_R.timestamp;
        if (detect_) {
            //thread_pool_->enqueue([&commonframe, this]() { detect_->pushInput(commonframe); }, -1);
            detect_->pushInput(commonframe);
        }
        // cv::imshow("left", img_L);
        // cv::imshow("right", img_R);
        // cv::waitKey(1);
    }
}
void WustBinocularNode::stopTimer() {
    {
        std::lock_guard<std::mutex> lk(timer_mtx_);
        timer_running_ = false;
    }
    timer_cv_.notify_one();
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
}
void WustBinocularNode::startTimer() {
    if (timer_running_)
        return;
    WUST_INFO("startTimer") << "starting timer";

    timer_running_ = true;

    double us_interval = 1e6 / static_cast<double>(fps);
    auto interval = std::chrono::microseconds(static_cast<int64_t>(us_interval));

    constexpr auto spin_margin = std::chrono::microseconds(200);

    timer_thread_ = std::thread([this, interval, spin_margin]() {
        auto next_time = std::chrono::steady_clock::now() + interval;
        auto last_time = std::chrono::steady_clock::now();

        while (true) {
            {
                std::unique_lock<std::mutex> lk(timer_mtx_);
                auto sleep_until = next_time - spin_margin;
                timer_cv_.wait_until(lk, sleep_until, [this]() { return !timer_running_; });
                if (!timer_running_)
                    break;
            }

            while (std::chrono::steady_clock::now() < next_time) {
                // busy‐wait
            }

            auto now = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(now - last_time).count();
            last_time = now;

            this->timerCallback();
            next_time += interval;
        }
    });
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
        if (timer_count_ < fps / 10) {
            timer_check_count++;
        }
        if (timer_check_count > 5) {
            // stopTimer();
            // startTimer();
            timer_check_count = 0;
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
    // initSingleCamera(
    //     "cameraR",
    //     use_video_R_,
    //     camera_R_,
    //     video_player_R_,
    //     camera_intrinsic_R_,
    //     camera_distortion_R_,
    //     video_alpha_R_,
    //     video_beta_R_
    // );
    initSingleCamera(
        "cameraL",
        use_video_L_,
        camera_L_,
        video_player_L_,
        camera_intrinsic_L_,
        camera_distortion_L_,
        video_alpha_L_,
        video_beta_L_
    );
}

void WustBinocularNode::initSingleCamera(
    const std::string& prefix,
    bool& use_video,
    std::unique_ptr<HikCamera>& camera,
    std::unique_ptr<VideoPlayer>& video_player,
    cv::Mat& intrinsic,
    cv::Mat& distortion,
    double& alpha,
    int& beta
) {
    std::string use_video_key = prefix + ".video_player.use";
    this->declare_parameter<bool>(use_video_key, true);
    WUST_INFO("initcamera") << prefix;
    this->get_parameter(use_video_key, use_video);

    if (use_video) {
        WUST_INFO("initcamera") << "use video";
        std::string path_key = prefix + ".video_player.path";
        std::string fps_key = prefix + ".video_player.fps";
        std::string start_frame_key = prefix + ".video_player.start_frame";
        std::string loop_key = prefix + ".video_player.loop";
        std::string alpha_key = prefix + ".video_player.alpha";
        std::string beta_key = prefix + ".video_player.beta";

        this->declare_parameter<std::string>(path_key, "");
        this->declare_parameter<int>(fps_key, 30);
        this->declare_parameter<int>(start_frame_key, 0);
        this->declare_parameter<bool>(loop_key, false);
        this->declare_parameter<double>(alpha_key, 1.0);
        this->declare_parameter<int>(beta_key, 0);

        std::string video_play_path;
        int video_play_fps, start_frame;
        bool loop;
        this->get_parameter(path_key, video_play_path);
        this->get_parameter(fps_key, video_play_fps);
        this->get_parameter(start_frame_key, start_frame);
        this->get_parameter(loop_key, loop);
        this->get_parameter(alpha_key, alpha);
        this->get_parameter(beta_key, beta);

        video_player =
            std::make_unique<VideoPlayer>(video_play_path, video_play_fps, start_frame, loop);
        //video_player->enableTriggerMode(true);
        video_player->setCallback([this](ImageFrame& frame) {
            if (!this->is_inited_) {
                return;
            }
            thread_pool_->enqueue([this, &frame]() {
                try {
                    this->frameCallback(frame);
                } catch (const std::exception& e) {
                    std::cerr << "Error in detect: " << e.what() << std::endl;
                }
            });
        });

    } else {
        std::string sn_key = prefix + ".target_sn";
        std::string rate_key = prefix + ".acquisition_frame_rate";
        std::string exposure_key = prefix + ".exposure_time";
        std::string gain_key = prefix + ".gain";
        std::string gamma_key = prefix + ".gamma";
        std::string bit_key = prefix + ".adc_bit_depth";
        std::string fmt_key = prefix + ".pixel_format";
        std::string rate_en_key = prefix + ".acquisitionFrameRateEnable";
        std::string rx_key = prefix + ".reverse_x";
        std::string ry_key = prefix + ".reverse_y";

        this->declare_parameter<std::string>(sn_key, "");
        this->declare_parameter<int>(rate_key, 250);
        this->declare_parameter<int>(exposure_key, 1000);
        this->declare_parameter<double>(gain_key, 16.9);
        this->declare_parameter<double>(gamma_key, 0.7);
        this->declare_parameter<std::string>(bit_key, "Bits_8");
        this->declare_parameter<std::string>(fmt_key, "BayerRG8");
        this->declare_parameter<bool>(rate_en_key, true);
        this->declare_parameter<bool>(rx_key, false);
        this->declare_parameter<bool>(ry_key, false);

        std::string sn, adc_bit_depth, pixel_format;
        int rate, exposure;
        double gain, gamma;
        bool rate_en, reverse_x, reverse_y;

        this->get_parameter(sn_key, sn);
        this->get_parameter(rate_key, rate);
        this->get_parameter(exposure_key, exposure);
        this->get_parameter(gain_key, gain);
        this->get_parameter(gamma_key, gamma);
        this->get_parameter(bit_key, adc_bit_depth);
        this->get_parameter(fmt_key, pixel_format);
        this->get_parameter(rate_en_key, rate_en);
        this->get_parameter(rx_key, reverse_x);
        this->get_parameter(ry_key, reverse_y);

        camera = std::make_unique<HikCamera>();
        if (!camera->initializeCamera(sn)) {
            WUST_ERROR("vision_logger") << prefix << " Camera initialization failed.";
            return;
        }

        camera->setParameters(
            rate,
            exposure,
            gain,
            gamma,
            adc_bit_depth,
            pixel_format,
            rate_en,
            reverse_x,
            reverse_y
        );
        camera->enableTrigger(TriggerType::Software, "Software", 0);
    }

    // 相机内参加载
    std::string info_key = prefix + ".camera_info_path";
    this->declare_parameter<std::string>(info_key, "");
    std::string info_path;
    this->get_parameter(info_key, info_path);
    WUST_DEBUG("CAMERA_INIT") << "CAMERA_INFO_PATH: " << info_path;
    YAML::Node config = YAML::LoadFile(info_path);
    std::vector<double> k = config["camera_matrix"]["data"].as<std::vector<double>>();
    std::vector<double> d = config["distortion_coefficients"]["data"].as<std::vector<double>>();
    assert(k.size() == 9);
    assert(d.size() == 5);

    cv::Mat K(3, 3, CV_64F), D(1, 5, CV_64F);
    std::memcpy(K.data, k.data(), 9 * sizeof(double));
    std::memcpy(D.data, d.data(), 5 * sizeof(double));

    intrinsic = K.clone();
    distortion = D.clone();
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(WustBinocularNode)