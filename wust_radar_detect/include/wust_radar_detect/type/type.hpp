#pragma once
#include "opencv2/opencv.hpp"
#include "yolos.hpp"
#include <chrono>
#include <functional>

struct CommonFrame {
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
};
class Car {
public:
    cv::Rect car_rect;
    yolo::Box car;
    yolo::BoxArray armors;
    cv::Point2f center;
    cv::Rect center_rect;
    int number = 0;
    int color = 1;
};
struct Cars {
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Car> cars;
};
struct imgframe {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
};
