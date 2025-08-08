#pragma once
#include "wust_radar_detect/type/type.hpp"
#include <optional>
#include <vector>
struct DetectDebug {
    std::optional<imgframe> imgframe_;
    std::optional<std::vector<Car>> cars;
    std::optional<std::chrono::steady_clock::time_point> detect_start;
    void reset() {
        imgframe_.reset();
        cars.reset();
    }
};

void showDebug(const DetectDebug& detect_debug);