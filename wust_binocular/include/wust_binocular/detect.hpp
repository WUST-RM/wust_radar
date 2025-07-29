#pragma once

#include "classify.hpp"
#include "wust_binocular/debug.hpp"
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

    bool load(std::memory_order m = std::memory_order_seq_cst) const noexcept {
        return v.load(m);
    }
    void store(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        v.store(b, m);
    }
    bool exchange(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        return v.exchange(b, m);
    }

    MovableAtomicBool(MovableAtomicBool&& o) noexcept: v(o.v.load(std::memory_order_relaxed)) {}
    MovableAtomicBool& operator=(MovableAtomicBool&& o) noexcept {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    MovableAtomicBool(const MovableAtomicBool&) = delete;
    MovableAtomicBool& operator=(const MovableAtomicBool&) = delete;
};
struct DetectConfig {
    int max_infer_threads;
    std::string config_path;
};

class Detect {
public:
    Detect(const DetectConfig& config);
    ~Detect();
    void pushInput(const CommonFrame& frame);
    size_t detect_finish_count_;

private:
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
    int max_infer_threads_;
    std::atomic<int> infer_running_count_ { 0 };
};