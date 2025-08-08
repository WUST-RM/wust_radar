#include "wust_radar_core/debug.hpp"
#include "wust_radar_core/utils.hpp"
constexpr double field_width_cm = 1500.0;
constexpr double field_height_cm = 2800.0;

cv::Mat DrawPointsOnImage(const cv::Mat& image_in, const RightCars& right_cars) {
    // 复制一份图像用于绘制
    cv::Mat image = image_in.clone();
    for (auto& car: right_cars.right_cars) {
        auto pt = car.uwb_point;
        int px = static_cast<int>((pt.x()) * 100.0 / field_height_cm * image.cols);
        int py = static_cast<int>((pt.y()) * 100.0 / field_width_cm * image.rows);

        // 如果Y轴方向需要反转，取消下面注释
        px = image.cols - px;
        py = image.rows - py;

        std::pair<cv::Scalar, std::string> color_and_name;
        switch (utils::carClass2ColorId(car.car_class).first) {
            case 0: // 蓝
                color_and_name.first = cv::Scalar(255, 0, 0); // BGR
                break;
            case 2: // 红
                color_and_name.first = cv::Scalar(0, 0, 255);
                break;
            case 1: // 绿
                color_and_name.first = cv::Scalar(255, 255, 255);
                break;
            default:
                color_and_name.first = cv::Scalar(128, 128, 128);
                break;
        }

        switch (utils::carClass2ColorId(car.car_class).second) {
            case 0:
                color_and_name.second = "unknow";
                break;
            case 1:
                color_and_name.second = "1";
                break;
            case 2:
                color_and_name.second = "2";
                break;
            case 3:
                color_and_name.second = "3";
                break;
            case 4:
                color_and_name.second = "4";
                break;
            case 6:
                color_and_name.second = "7";
                break;
            default:
                color_and_name.second = "?";
                break;
        }

        // 边界检查
        if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
            // 画圆点
            cv::circle(image, cv::Point(px, py), 5, color_and_name.first, -1);

            // 左上角绘制文字（偏移一点避免和圆重叠）
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int thickness = 1;
            cv::putText(
                image,
                color_and_name.second,
                cv::Point(px - 10, py - 10),
                font_face,
                font_scale,
                color_and_name.first,
                thickness
            );
        }
    }

    return image;
}
