#include "wust_binocular/detect.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <wust_utils/logger.hpp>
#include <yaml-cpp/yaml.h>

#define MAX_CARS 12
Detect::~Detect() {
    WUST_INFO("detect") << "Detect destructor called.";
    //infers.clear();
    if (thread_pool_) {
        thread_pool_->waitUntilEmpty();
        thread_pool_.reset();
    }
}
Detect::Detect(const DetectConfig& cfg) {
    max_infer_threads_ = cfg.max_infer_threads;
    min_free_mem_ratio_ = cfg.min_free_mem_ratio;
    std::ifstream ac_status("/sys/class/power_supply/AC/online");
    if (!ac_status.is_open()) {
        WUST_ERROR("detect_init") << "无法读取电源状态文件";
        std::exit(EXIT_FAILURE);
    }

    int status;
    ac_status >> status;
    if (status == 1) {
        WUST_INFO("detect_init") << "正在使用电源（已插入电源适配器）";
    } else if (status == 0) {
        WUST_ERROR("detect_init") << "使用电池供电,trt 引擎无法分配";
        std::exit(EXIT_FAILURE);
    } else {
        WUST_ERROR("detect_init") << "电源状态未知";
        std::exit(EXIT_FAILURE);
    }
    WUST_INFO("detect_init") << "Checking CUDA with nvidia-smi...";
    if (system("nvidia-smi") == 0) {
        WUST_INFO("detect_init") << "CUDA is available.";
    } else {
        WUST_ERROR("detect_init") << "CUDA is not available. Exiting.";
        std::exit(EXIT_FAILURE);
    }

    auto config_path = cfg.config_path;

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
            "--Shape=960x960 "
            "--input_name=images"
        );
    } else {
        WUST_INFO("detect_init") << "Load yolo engine!";
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
        WUST_INFO("detect_init") << "Load armor_yolo engine!";
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
        WUST_INFO("detect_init") << "Load classify engine!";
    }

    std::cout << "yolo_path: " << yolo_path << "\n";
    std::cout << "armor_path: " << armor_path << "\n";
    std::cout << "classify_path: " << classify_path << "\n";

    infers.reserve(max_infer_threads_);
    for (size_t i = 0; i < max_infer_threads_; i++) {
        auto infer = std::make_unique<Inf>();
        infer->yolo = yolo::load(yolo_path, yolo::Type::V5, 0.6f, 0.45f);
        infer->armor_yolo = yolo::load(armor_path, yolo::Type::V5, 0.4f, 0.45f);
        infer->classifier = classify::load(classify_path, classify::Type::densenet121);
        if (!infer->yolo || !infer->armor_yolo || !infer->classifier) {
            WUST_ERROR("detect_init") << "Load infer failed!"
                                      << "index:" << i;
            //infer_status_[i].store(true);
            continue;
        }
        infers.push_back(std::move(infer));
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        WUST_DEBUG("detect_init") << "Free GPU memory:" << free_mem / 1024.0 / 1024.0 << "MB"
                                  << "Total GPU memory:" << total_mem / 1024.0 / 1024.0 << "MB";
        double free_mem_ratio = static_cast<double>(free_mem) / static_cast<double>(total_mem);
        if (free_mem_ratio < min_free_mem_ratio_) {
            WUST_WARN("detect_init") << "GPU memory is not enough!"
                                     << "Free GPU memory:" << free_mem_ratio * 100 << "%";
            WUST_INFO("detect_init") << "Cut remaining infer";
            break;
        }
        WUST_MAIN("detect_init") << "Load infer success!"
                                 << "index:" << i;
    }
    infer_status_.reserve(infers.size());
    for (size_t i = 0; i < infers.size(); ++i) {
        infer_status_.emplace_back(false);
    }
    for (size_t i = 0; i < infers.size(); ++i) {
        infer_released_.emplace_back(false);
    }
    WUST_MAIN("detect_init") << "Load infers:" << infers.size() << " success!";

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
cv::Mat letterbox(const cv::Mat& img, int target_size = 224) {
    int w = img.cols, h = img.rows;
    float scale = std::min(float(target_size) / w, float(target_size) / h);
    int new_w = int(w * scale), new_h = int(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;

    cv::Mat output;
    cv::copyMakeBorder(
        resized,
        output,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );
    return output;
}
cv::Mat resizeNoPadding(const cv::Mat& img, int target_size = 224) {
    cv::Mat output;
    cv::resize(img, output, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);
    return output;
}

std::vector<Car> Detect::detect(
    const CommonFrame& frame,
    const std::unique_ptr<Inf>& infer,
    DetectDebug& detect_debug
) {
    cv::Mat img = frame.image_L;

    if (img.empty()) {
        WUST_ERROR("detect") << "Empty input image!";
        return {};
    }
    if (debug) {
        if (!detect_debug.imgframe_)
            detect_debug.imgframe_.emplace();
        img.convertTo(detect_debug.imgframe_->img, -1, 1, 0);
        detect_debug.imgframe_->timestamp = frame.timestamp_L;
        detect_debug.detect_start.emplace(std::chrono::steady_clock::now());
    }
    yolo::Image image(img.data, img.cols, img.rows);
    if (!infer->yolo) {
        WUST_ERROR("detect") << "yolo is null";
        return {};
    }
    auto result = infer->yolo->forward(image);
    if (result.size() == 0) {
        //WUST_INFO("detect") << "No Car!";
        return {};
    } else if (result.size() > MAX_CARS) {
        WUST_INFO("detect") << "Too Many Car!" << result.size() << " > " << MAX_CARS;
        return {};
    }

    std::vector<yolo::Image> images;
    std::vector<cv::Mat> car_imgs;
    std::vector<Car> cars;
    for (auto& box: result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;
            cars.push_back(car);
        }
    }

    for (auto& car: cars) {
        auto temp_rect = cv::Rect(
            car.car.left,
            car.car.top,
            car.car.right - car.car.left,
            car.car.bottom - car.car.top
        );
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
        auto car_img = img(temp_car_rect);
        car_imgs.push_back(car_img.clone());
        car.car_rect = temp_car_rect;
    }

    if (!infer->armor_yolo) {
        if (debug) {
            detect_debug.cars = cars;
        }
        WUST_ERROR("detect") << "armor_yolo is null";
        return cars;
    }
    for (auto& car_img: car_imgs) {
        auto image = yolo::Image(car_img.data, car_img.cols, car_img.rows);
        images.push_back(image);
    }

    auto armor_boxes = infer->armor_yolo->forwards(images);
    bool has_armor = false;
    for (int i = 0; i < armor_boxes.size(); i++) {
        if (armor_boxes[i].size() == 0) {
            continue;
        } else {
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }
    if (!has_armor) {
        //WUST_INFO("detect") << "No Armor!";
        if (debug) {
            detect_debug.cars = cars;
        }
        return cars;
    }
    if (!infer->classifier) {
        if (debug) {
            detect_debug.cars = cars;
        }
        WUST_ERROR("detect") << "classifier is null";
        return cars;
    }

    std::vector<cv::Mat> armor_imgs;
    std::vector<classify::Image> armor_images;

    for (auto& car: cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor: car.armors) {
            cv::Rect rect_img_1(
                armor.left + car.car.left,
                armor.top + car.car.top,
                armor.right - armor.left,
                armor.bottom - armor.top
            );
            cv::Rect rect_img = getSafeRect(img, rect_img_1);
            auto armor_img = img(rect_img);
            armor_imgs.push_back(armor_img.clone());
        }
    }
    for (auto& armor_img: armor_imgs) {
        auto image = classify::Image(armor_img.data, armor_img.cols, armor_img.rows);
        armor_images.push_back(image);
    }
    auto armor_result = infer->classifier->forwards(armor_images);

    for (auto& car: cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        cv::Rect max_rect;
        float max_confidence = 0;
        std::vector<yolo::Box> fallback_armors;
        for (auto& armor: car.armors) {
            armor.class_label = armor_result[0];
            armor_result.erase(armor_result.begin());

            if (armor.class_label != 0) {
                if (armor.confidence > max_confidence) {
                    max_rect = cv::Rect(
                        armor.left + car.car.left,
                        armor.top + car.car.top,
                        armor.right - armor.left,
                        armor.bottom - armor.top
                    );
                    max_confidence = armor.confidence;
                    car.number = armor.class_label;
                }
            } else {
                fallback_armors.push_back(armor);
            }
        }
        if (car.number == 0 && !fallback_armors.empty()) {
            yolo::Box* best = nullptr;
            float best_conf = -1.0f;
            for (auto& armor: fallback_armors) {
                if (armor.confidence > best_conf) {
                    best = &armor;
                    best_conf = armor.confidence;
                }
            }

            if (best) {
                max_rect = cv::Rect(
                    best->left + car.car.left,
                    best->top + car.car.top,
                    best->right - best->left,
                    best->bottom - best->top
                );
                max_confidence = best->confidence;
                car.number = 0;
            }
        }
        if (max_confidence == 0) {
            continue;
        }
        auto safe_rect = getSafeRect(img, max_rect);
        auto max_mat = img(safe_rect);

        car.color = getColor(max_mat);
        car.center =
            cv::Point2f(max_rect.x + max_rect.width / 2.0f, max_rect.y + max_rect.height / 2.0f);
    }
    if (debug) {
        detect_debug.cars = cars;
    }
    return cars;
}

void Detect::pushInput(const CommonFrame& frame) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;

    int ctx_num = std::count(infer_released_.begin(), infer_released_.end(), false);
    double used_ratio = (double)used_mem / total_mem;
    double extra_margin = used_ratio * ctx_num * 0.1;

    double free_mem_ratio = static_cast<double>(free_mem) / total_mem;

    if (free_mem_ratio >= min_free_mem_ratio_ + extra_margin) {
        for (size_t i = 0; i < infers.size(); ++i) {
            if (infer_released_[i]) {
                infers[i] = std::make_unique<Inf>(); 
                infers[i]->yolo = yolo::load(yolo_path, yolo::Type::V5, 0.6f, 0.45f);
                infers[i]->armor_yolo = yolo::load(armor_path, yolo::Type::V5, 0.4f, 0.45f);
                infers[i]->classifier = classify::load(classify_path, classify::Type::densenet121);
                if (infers[i] && infers[i]->yolo && infers[i]->armor_yolo && infers[i]->classifier) {
                    infer_released_[i] = false;
                    infer_status_[i].store(false);
                    WUST_INFO("pushInput") << "Restored infer index: " << i;
                } else {
                    std::cerr << "Failed to restore infer[" << i << "]\n";
                }
            }
        }
    }

    for (size_t i = 0; i < infer_status_.size(); ++i) {
        if (!infer_status_[i].load() && !infer_released_[i]) {
            if (free_mem_ratio < min_free_mem_ratio_) {
                WUST_WARN("pushInput") << "GPU memory is not enough!"
                                       << "Free GPU memory:" << free_mem_ratio * 100 << "%";

                // 至少保留一个可用 infers
                size_t active_count = 0;
                for (bool released : infer_released_) {
                    if (!released) ++active_count;
                }

                for (size_t j = i; j < infer_status_.size(); ++j) {
                    if (!infer_status_[j].load() && !infer_released_[j]) {
                        if (active_count <= 1) {
                            break;  // 不再释放，至少保留一个
                        }

                        infer_status_[j].store(true);
                        infers[j].reset();  // 释放资源
                        infer_released_[j] = true;
                        WUST_INFO("pushInput")
                            << "Cut infer index:" << j << " due to low GPU memory";
                        break;
                    }
                }
                break;
            }

            infer_status_[i].store(true);

            thread_pool_->enqueue([this, i, frame]() {
                try {
                    if (!infers[i]) {
                        std::cerr << "Fatal: infers[" << i << "] is nullptr\n";
                        infer_status_[i].store(false);
                        return;
                    }
                    if (!infers[i]->yolo || !infers[i]->armor_yolo || !infers[i]->classifier) {
                        std::cerr << "Fatal: infers[" << i << "] contains nullptr components\n";
                        infer_status_[i].store(false);
                        return;
                    }

                    DetectDebug detect_debug;
                    auto cars = this->detect(frame, infers[i], detect_debug);
                    if (callback_) {
                        callback_(frame, cars, detect_debug);
                    }

                } catch (const std::exception& e) {
                    std::cerr << "Error in detect: " << e.what() << std::endl;
                }
                infer_status_[i].store(false);
                detect_finish_count_++;
            });

            return;
        }
    }
}
