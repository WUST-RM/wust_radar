#pragma once

#include "classify.hpp"
#include "wust_binocular/type/type.hpp"
#include "wust_utils/ThreadPool.h"
#include "yolos.hpp"
#include <memory>
struct Inf {
    std::shared_ptr<yolo::Infer> yolo;
    std::shared_ptr<yolo::Infer> armor_yolo;
    std::shared_ptr<classify::Infer> classifier;
};
struct MovableAtomicBool {
    std::atomic<bool> v;
    explicit MovableAtomicBool(bool b = false) noexcept: v(b) {}

    // —— 转发 atomic<bool> 的接口 ——
    bool load(std::memory_order m = std::memory_order_seq_cst) const noexcept {
        return v.load(m);
    }
    void store(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        v.store(b, m);
    }
    bool exchange(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        return v.exchange(b, m);
    }
    // （如果你用到了其他 atomic 操作，也一并封装进来）

    // —— 可移动语义 ——
    MovableAtomicBool(MovableAtomicBool&& o) noexcept: v(o.v.load(std::memory_order_relaxed)) {}
    MovableAtomicBool& operator=(MovableAtomicBool&& o) noexcept {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    // 禁用拷贝，防止误用
    MovableAtomicBool(const MovableAtomicBool&) = delete;
    MovableAtomicBool& operator=(const MovableAtomicBool&) = delete;
};
struct DetectConfig {
    int max_infer_threads;
};
struct DetectDebug {
    std::optional<imgframe> imgframe_;
    std::optional<std::vector<Car>> cars;
};
class Detect {
public:
    Detect(const DetectConfig& config);
    void pushInput(CommonFrame& frame);
    void showDebug(const DetectDebug& detect_debug);
    size_t detect_finish_count_;

private:
    void detect42mm(const cv::Mat& image);
    void
    detect(const CommonFrame& frame, const std::unique_ptr<Inf>& infer, DetectDebug& detect_debug);
    std::vector<std::unique_ptr<Inf>> infers;
    std::vector<MovableAtomicBool> infer_status_;
    std::string yolo_path;
    std::string armor_path;
    std::string classify_path;
    bool debug = true;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::atomic<int> next_infer_id_ { 0 };
    int max_infer_threads_ = 4;
    std::atomic<int> infer_running_count_ { 0 };
};