#pragma once

#include "classify.hpp"
#include "wust_radar_detect/debug.hpp"
#include "wust_radar_detect/type/type.hpp"
#include "wust_utils/ThreadPool.h"
#include "wust_utils/adaptive_resource_pool.hpp"
#include "yolos.hpp"
#include <memory>
using DetectCallback =
    std::function<void(const CommonFrame&, const Cars&, DetectDebug&)>;
struct Inf {
    std::shared_ptr<yolo::Infer> yolo;
    std::shared_ptr<yolo::Infer> armor_yolo;
    std::shared_ptr<classify::Infer> classifier;
};

struct DetectConfig {
    int max_infer_threads;
    std::string config_path;
    double min_free_mem_ratio;
};

class Detect {
public:
    Detect(const DetectConfig& config);
    ~Detect();
    void pushInput(const CommonFrame& frame);
    size_t detect_finish_count_;
    void setCallback(const DetectCallback& cb) {
        callback_ = cb;
    }
    std::unique_ptr<AdaptiveResourcePool<Inf>> resource_pool_;

private:
    Cars detect(const CommonFrame& frame, Inf* infer, DetectDebug& detect_debug);
    std::string yolo_path;
    std::string armor_path;
    std::string classify_path;
    bool debug = true;
    std::shared_ptr<ThreadPool> thread_pool_;
    int max_infer_threads_;
    double min_free_mem_ratio_ = 0.5;
    std::atomic<int> infer_running_count_ { 0 };
    DetectCallback callback_;
};