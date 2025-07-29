#include "wust_binocular/debug.hpp"
#include "fmt/format.h"
void showDebug(const DetectDebug& detect_debug) {
    if (!detect_debug.imgframe_ || detect_debug.imgframe_->img.empty()) {
        std::cout << "no img" << std::endl;
        return;
    }
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    cv::Mat debug_img = detect_debug.imgframe_->img.clone();
    //cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
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
            cv::rectangle(debug_img, car.car_rect, col, 4);
            for (auto& arm: car.armors) {
                cv::Rect r(
                    arm.left + car.car.left,
                    arm.top + car.car.top,
                    arm.right - arm.left,
                    arm.bottom - arm.top
                );
                cv::rectangle(debug_img, r, { 255, 255, 255 }, 2);
                cv::putText(
                    debug_img,
                    std::to_string(arm.class_label),
                    cv::Point(r.x, r.y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }
            std::string top_string = std::to_string(car.car.confidence);
            cv::putText(
                debug_img,
                top_string,
                cv::Point(car.car.left, car.car.top),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                col,
                2
            );
            int baseline = 0;
            std::string bottom_string = std::to_string(car.number);
            cv::Size text_size =
                cv::getTextSize(bottom_string, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
            cv::putText(
                debug_img,
                bottom_string,
                cv::Point(car.car.right, car.car.bottom + text_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                2,
                col,
                5
            );
        }
    }

    double total_dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                          now - detect_debug.imgframe_->timestamp
                      )
                          .count()
        * 1000;
    std::string total_text = fmt::format("Total Time: {:.2f} ms", total_dt);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(total_text, cv::FONT_HERSHEY_SIMPLEX, 2.0, 5, &baseline);
    int line_height = text_size.height + 10;

    cv::Point org1(10, 100);
    cv::Point org2(10, org1.y + line_height);
    cv::putText(
        debug_img,
        total_text,
        org1,
        cv::FONT_HERSHEY_SIMPLEX,
        2.0,
        { 0, 0, 0 },
        20,
        cv::LINE_AA
    );
    cv::putText(
        debug_img,
        total_text,
        org1,
        cv::FONT_HERSHEY_SIMPLEX,
        2.0,
        { 255, 255, 255 },
        5,
        cv::LINE_AA
    );

    if (detect_debug.detect_start) {
        std::chrono::duration<double> time_used =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                now - detect_debug.detect_start.value()
            );
        double time_ms = time_used.count() * 1000;
        std::string cost_text = fmt::format("Detect Time: {:.2f} ms", time_ms);
        cv::putText(
            debug_img,
            cost_text,
            org2,
            cv::FONT_HERSHEY_SIMPLEX,
            2.0,
            { 0, 0, 0 },
            20,
            cv::LINE_AA
        );
        cv::putText(
            debug_img,
            cost_text,
            org2,
            cv::FONT_HERSHEY_SIMPLEX,
            2.0,
            { 255, 255, 255 },
            5,
            cv::LINE_AA
        );
    }

    // 显示
    cv::resizeWindow("detect", 800, 600);
    cv::namedWindow("detect", cv::WINDOW_NORMAL);
    cv::imshow("detect", debug_img);
    cv::waitKey(1);
}