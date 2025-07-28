#include "wust_binocular/detect.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <wust_utils/logger.hpp>
#include <yaml-cpp/yaml.h>
#define MAX_CARS 12
Detect::Detect(const DetectConfig& cfg) {
    max_infer_threads_ = cfg.max_infer_threads;
    std::cout << "Checking CUDA with nvidia-smi...\n";
    if (system("nvidia-smi") == 0) {
        WUST_INFO("detect") << "CUDA is available.";
    } else {
        WUST_ERROR("detect") << "CUDA is not available. Exiting.";
        std::exit(1);
    }

    std::string config_path = "/home/hy/wust_radar/src/wust_binocular/config/detect_params.yaml";

    // 使用 yaml-cpp 加载配置文件
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config) {
        throw std::runtime_error("Failed to load config file: " + config_path);
    }

    try {
        yolo_path = config["yolo_path"].as<std::string>();
        armor_path = config["armor_path"].as<std::string>();
        classify_path = config["classify_path"].as<std::string>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(std::string("YAML parse error: ") + e.what());
    }

    // 检查 yolo engine 文件是否存在，不存在则调用转换脚本
    std::ifstream file1(yolo_path);
    if (!file1.good()) {
        system(
            "python3 /home/hy/wust_radar/src/wust_binocular/utils/onnx2trt.py "
            "--onnx=/home/hy/wust_radar/src/wust_binocular/model/ONNX/RM2024.onnx "
            "--saveEngine=/home/hy/wust_radar/src/wust_binocular/model/TensorRT/"
            "yolo.engine "
            "--minBatch 1 "
            "--optBatch 1 "
            "--maxBatch 2 "
            "--Shape=640x640 "
            "--input_name=images"
        );
    } else {
        WUST_INFO("detect") << "Load yolo engine!";
    }

    std::ifstream file2(armor_path);
    if (!file2.good()) {
        system(
            "python3 /home/hy/wust_radar/src/wust_binocular/utils/onnx2trt.py "
            "--onnx=/home/hy/wust_radar/src/wust_binocular/model/ONNX/"
            "armor_yolo.onnx "
            "--saveEngine=/home/hy/wust_radar/src/wust_binocular/model/TensorRT/"
            "armor_yolo.engine "
            "--minBatch 1 "
            "--optBatch 5 "
            "--maxBatch 12 "
            "--Shape=96x96 "
            "--input_name=images"
        );
    } else {
        WUST_INFO("detect") << "Load armor_yolo engine!";
    }

    std::ifstream file3(classify_path);
    if (!file3.good()) {
        system(
            "python3 /home/hy/wust_radar/src/wust_binocular/utils/onnx2trt.py "
            "--onnx=/home/hy/wust_radar/src/wust_binocular/model/ONNX/"
            "classify.onnx "
            "--saveEngine=/home/hy/wust_radar/src/wust_binocular/model/TensorRT/"
            "classify.engine "
            "--minBatch 1 "
            "--optBatch 10 "
            "--maxBatch 20 "
            "--Shape=224x224 "
            "--input_name=input"
        );
    } else {
        WUST_INFO("detect") << "Load classify engine!";
    }

    std::cout << "yolo_path: " << yolo_path << "\n";
    std::cout << "armor_path: " << armor_path << "\n";
    std::cout << "classify_path: " << classify_path << "\n";
    for (size_t i = 0; i < max_infer_threads_; i++) {
        auto infer = std::make_unique<Inf>();

        infer->yolo = yolo::load(yolo_path, yolo::Type::V5, 0.6f, 0.45f);
        infer->armor_yolo = yolo::load(armor_path, yolo::Type::V5, 0.4f, 0.45f);
        infer->classifier = classify::load(classify_path, classify::Type::densenet121);
        infers.push_back(std::move(infer));
        WUST_INFO("detect") << "Load infer success!";
    }
    infer_status_.reserve(max_infer_threads_);
    for (size_t i = 0; i < max_infer_threads_; ++i) {
        infer_status_.emplace_back(false);
    }

    // this->classifier = classify::load(classify_path, classify::Type::densenet121);
    // WUST_INFO("detect") << "Load classify engine success!";

    // this->armor_yolo = yolo::load(armor_path, yolo::Type::V5, 0.4f, 0.45f);
    // WUST_INFO("detect") << "Load armor_yolo engine success!";

    // this->yolo = yolo::load(yolo_path, yolo::Type::V5, 0.65f, 0.45f);
    // WUST_INFO("detect") << "Load yolo engine success!";
    thread_pool_ = std::make_unique<ThreadPool>(std::thread::hardware_concurrency() * 2);
    detect_finish_count_ = 0;
}
int getColor(cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat blueMinusRed = channels[0] - channels[2];
    cv::Mat redMinusBlue = channels[2] - channels[0];
    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);
    cv::Scalar avgGreen = cv::mean(channels[1]);
    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0;
    } else {
        return 2;
    }
}

bool isRectInside(const cv::Rect& small, const cv::Rect& big) {
    bool topLeftInside = big.contains(small.tl());
    bool topRightInside = big.contains(cv::Point(small.x + small.width, small.y));
    bool bottomLeftInside = big.contains(cv::Point(small.x, small.y + small.height));
    bool bottomRightInside = big.contains(cv::Point(small.x + small.width, small.y + small.height));
    return (topLeftInside && topRightInside && bottomLeftInside && bottomRightInside);
}

bool isBoxInside(const yolo::Box& small, const yolo::Box& big) {
    cv::Rect small_rect(small.left, small.top, small.right - small.left, small.bottom - small.top);
    cv::Rect big_rect(big.left, big.top, big.right - big.left, big.bottom - big.top);
    return isRectInside(small_rect, big_rect);
}
cv::Rect getSafeRect(const cv::Mat& image, const cv::Rect& rect) {
    int x = std::max(0, rect.x);
    int y = std::max(0, rect.y);
    int width = std::min(rect.width, image.cols - x);
    int height = std::min(rect.height, image.rows - y);
    width = std::max(0, width);
    height = std::max(0, height);

    return cv::Rect(x, y, width, height);
}

// In Detect.cpp

void Detect::detect(
    const CommonFrame& frame,
    const std::unique_ptr<Inf>& infer,
    DetectDebug& detect_debug
) {
    if (!infer->yolo || !infer->classifier || !infer->armor_yolo) {
        WUST_ERROR("detect") << "infer is null";
        return;
    }

    cv::Mat img = frame.image_L;

    if (img.empty()) {
        WUST_ERROR("detect") << "Empty input image!";
        return;
    }
    if (debug) {
        img.convertTo(detect_debug.imgframe_->img, -1, 1, 0);
        auto now = std::chrono::steady_clock::now();
        detect_debug.imgframe_->timestamp = now;
    }
    // 1. YOLO 车检测
    yolo::Image image(img.data, img.cols, img.rows);
    auto result = infer->yolo->forward(image);
    if (result.empty()) {
        WUST_INFO("detect") << "No Car!";

        return;
    } else if (result.size() > MAX_CARS) {
        WUST_INFO("detect") << "Too Many Car!" << result.size();
        return;
    }

    // 2. 筛选出车框并提取 ROI
    std::vector<Car> cars;
    std::vector<cv::Mat> car_imgs;
    for (auto& box: result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;
            cv::Rect rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
            car.car_rect = getSafeRect(img, rect);
            if (car.car_rect.area() > 0) {
                car_imgs.push_back(img(car.car_rect).clone());
                cars.push_back(car);
            }
        }
    }

    // 3. 装甲板检测
    std::vector<yolo::Image> armor_batch;
    for (auto& mat: car_imgs) {
        armor_batch.emplace_back(mat.data, mat.cols, mat.rows);
    }
    auto armor_boxes = infer->armor_yolo->forwards(armor_batch);

    bool has_armor = false;
    for (size_t i = 0; i < armor_boxes.size() && i < cars.size(); ++i) {
        if (!armor_boxes[i].empty()) {
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }
    if (!has_armor) {
        WUST_INFO("detect") << "No Armor!";
        if (debug) {
            detect_debug.cars = cars;
        }
        return;
    }

    // 4. 装甲分类
    std::vector<classify::Image> cls_batch;
    for (size_t i = 0; i < cars.size(); ++i) {
        for (auto& arm: cars[i].armors) {
            cv::Rect r_img(
                arm.left + cars[i].car.left,
                arm.top + cars[i].car.top,
                arm.right - arm.left,
                arm.bottom - arm.top
            );
            cv::Rect safe = getSafeRect(img, r_img);
            if (safe.area() > 0) {
                cls_batch.emplace_back(img(safe).data, safe.width, safe.height);
            }
        }
    }
    auto cls_results = infer->classifier->forwards(cls_batch);

    // 5. 将分类结果贴回 cars，并计算每辆车的最终装甲方位、颜色等
    size_t idx = 0;
    for (auto& car: cars) {
        for (auto& arm: car.armors) {
            if (idx < cls_results.size()) {
                arm.class_label = cls_results[idx++];
            }
        }
        // 选置信度最高的装甲，并计算中心、颜色 etc.
        float max_conf = 0;
        cv::Rect best_rect;
        for (auto& arm: car.armors) {
            if (arm.confidence > max_conf) {
                max_conf = arm.confidence;
                best_rect = cv::Rect(
                    arm.left + car.car.left,
                    arm.top + car.car.top,
                    arm.right - arm.left,
                    arm.bottom - arm.top
                );
                car.number = arm.class_label;
            }
        }
        if (max_conf > 0) {
            cv::Rect safe = getSafeRect(img, best_rect);
            auto max_mat = img(safe);
            car.color = getColor(max_mat);
            car.center = cv::Point2f(
                best_rect.x + best_rect.width / 2.f,
                best_rect.y + best_rect.height / 2.f
            );
        }
    }
    if (debug) {
        detect_debug.cars = cars;
    }
}

void Detect::showDebug(const DetectDebug& detect_debug) {
    if (detect_debug.imgframe_ && detect_debug.imgframe_->img.empty()) {
        return;
    }
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            now - detect_debug.imgframe_->timestamp
        );
    double time_ms = time_used.count() * 1000;
    cv::Mat debug_img = detect_debug.imgframe_->img.clone();
    if (detect_debug.cars) {
        const auto& cars = *detect_debug.cars;
        for (auto& car: cars) {
            cv::Scalar col;
            switch (car.color) {
                case 0:
                    col = { 255, 0, 0 };
                    break; // blue
                case 2:
                    col = { 0, 0, 255 };
                    break; // red
                default:
                    col = { 255, 255, 255 }; // white
            }
            cv::rectangle(debug_img, car.car_rect, col, 2);
            for (auto& arm: car.armors) {
                cv::Rect r(
                    arm.left + car.car.left,
                    arm.top + car.car.top,
                    arm.right - arm.left,
                    arm.bottom - arm.top
                );
                cv::rectangle(debug_img, r, { 255, 255, 255 }, 1);
                cv::putText(
                    debug_img,
                    std::to_string(arm.class_label),
                    cv::Point(r.x, r.y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    col,
                    2
                );
            }
            // 绘制置信度
            cv::putText(
                debug_img,
                std::to_string(car.car.confidence),
                cv::Point(car.car.left, car.car.top),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                col,
                2
            );
        }
    }

    // 绘制性能文字
    std::string text = "Detect Time: " + std::to_string(time_ms) + " ms";
    cv::Point org(10, 100);
    cv::putText(debug_img, text, org, cv::FONT_HERSHEY_SIMPLEX, 2.0, { 0, 0, 0 }, 7, cv::LINE_AA);
    cv::putText(
        debug_img,
        text,
        org,
        cv::FONT_HERSHEY_SIMPLEX,
        2.0,
        { 50, 255, 50 },
        5,
        cv::LINE_AA
    );

    // 显示
    cv::resizeWindow("detect", 800, 600);
    cv::namedWindow("detect", cv::WINDOW_NORMAL);
    cv::imshow("detect", debug_img);
    int key = cv::waitKey(1);
    if (key == 'r')
        debug = !debug;
}

void Detect::pushInput(CommonFrame& frame) {
    for (size_t i = 0; i < infer_status_.size(); ++i) {
        if (!infer_status_[i].load()) {
            infer_status_[i].store(true);

            thread_pool_->enqueue([this, i, frame = std::move(frame)]() {
                try {
                    DetectDebug detect_debug;
                    this->detect(frame, infers[i], detect_debug);
                    if (debug) {
                        showDebug(detect_debug);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error in detect: " << e.what() << std::endl;
                }

                infer_status_[i].store(false);
            });
            detect_finish_count_++;
            return;
        }
    }

    //std::cerr << "No free infer slots available. Frame discarded." << std::endl;
}
void Detect::detect42mm(const cv::Mat& image) {
    cv::namedWindow("Binary Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Binary Image", 800, 600);
    // 假设 image 是 CV_8UC3 的 BGR 图像（OpenCV 默认是 BGR，不是 RGB）
    cv::Mat green_channel;

    // 提取 G 通道 (通道索引为1)
    cv::extractChannel(image, green_channel, 1);

    cv::Mat binary_img;
    cv::threshold(green_channel, binary_img, 150, 255, cv::THRESH_BINARY);

    cv::imshow("Binary Image", binary_img);
    cv::waitKey(1);
    // 你后续处理就用 binary_img
}
