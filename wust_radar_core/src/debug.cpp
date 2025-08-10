#include "wust_radar_core/debug.hpp"
#include "wust_radar_core/utils.hpp"
constexpr double field_width_cm = 1500.0;
constexpr double field_height_cm = 2800.0;
void drawFilledStar(cv::Mat& img, cv::Point center, int radius, cv::Scalar color) {
    std::vector<cv::Point> pts(10);
    double angle = CV_PI / 2; // 90度起始角度，朝上
    double step = 2 * CV_PI / 5; // 72度

    for (int i = 0; i < 5; ++i) {
        // 外顶点
        pts[2 * i] = cv::Point(
            static_cast<int>(center.x + radius * cos(angle + i * step)),
            static_cast<int>(center.y - radius * sin(angle + i * step))
        );
        // 内顶点，半径大约为外顶点的0.4倍
        pts[2 * i + 1] = cv::Point(
            static_cast<int>(center.x + radius * 0.4 * cos(angle + i * step + step / 2)),
            static_cast<int>(center.y - radius * 0.4 * sin(angle + i * step + step / 2))
        );
    }

    const cv::Point* pts_ptr = pts.data();
    int npts = static_cast<int>(pts.size());
    cv::fillPoly(img, &pts_ptr, &npts, 1, color, cv::LINE_AA);
}

// 画圆环
void drawRing(cv::Mat& img, cv::Point center, int radius, cv::Scalar color, int thickness = 2) {
    cv::circle(img, center, radius, color, thickness, cv::LINE_AA);
}
cv::Mat DrawPointsOnImage(const cv::Mat& image_in, const FinalCars& cars) {
    cv::Mat image = image_in.clone();

    // 速度箭头最大长度，单位像素
    const int max_arrow_length_px = 50;

    for (auto& car: cars.final_cars) {
        auto pt = car.uwb_point;
        auto vel = car.uwb_velocity;

        // 点像素坐标
        int px = static_cast<int>((pt.x()) * 100.0 / field_height_cm * image.cols);
        int py = static_cast<int>((pt.y()) * 100.0 / field_width_cm * image.rows);
        px = image.cols - px;
        py = image.rows - py;

        // 速度像素增量
        int vx_px = static_cast<int>((vel.x()) * 100.0 / field_height_cm * image.cols);
        int vy_px = static_cast<int>((vel.y()) * 100.0 / field_width_cm * image.rows);
        // y方向翻转（图像坐标系y向下）
        vx_px = -vx_px;
        vy_px = -vy_px;

        // 限制速度箭头长度
        double length = std::sqrt(vx_px * vx_px + vy_px * vy_px);
        if (length > max_arrow_length_px && length > 1e-6) {
            double scale = max_arrow_length_px / length;
            vx_px = static_cast<int>(vx_px * scale);
            vy_px = static_cast<int>(vy_px * scale);
        }

        cv::Scalar color;
        switch (utils::carClass2ColorId(car.car_class).first) {
            case 0: color = cv::Scalar(255, 0, 0); break;   // 蓝色
            case 2: color = cv::Scalar(0, 0, 255); break;   // 红色
            default: color = cv::Scalar(128, 128, 128); break;
        }

        std::string label;
        switch (utils::carClass2ColorId(car.car_class).second) {
            case 0: label = "unknow"; break;
            case 1: label = "1"; break;
            case 2: label = "2"; break;
            case 3: label = "3"; break;
            case 4: label = "4"; break;
            case 6: label = "7"; break;
            default: label = "?"; break;
        }

        int star_radius = 5;
        int ring_radius = 3;
        int ring_thickness = 2;
        int circle_radius = 3;

        if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
            // 根据状态绘制形状
            switch (car.state) {
                case CarState::ACTIVE:
                    drawFilledStar(image, cv::Point(px, py), star_radius, color); // 五角星
                    break;
                case CarState::NEEDGUESS:
                case CarState::GUESSING:
                    drawRing(image, cv::Point(px, py), ring_radius, color, ring_thickness);
                    break;
                default:
                    cv::circle(image, cv::Point(px, py), circle_radius, color, -1);
                    break;
            }

            // 绘制速度箭头
            cv::Point start_point(px, py);
            cv::Point end_point(px + vx_px, py + vy_px);
            cv::arrowedLine(image, start_point, end_point, color, 1, cv::LINE_AA, 0, 0.3);

            // 绘制标签
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int thickness = 1;
            cv::putText(image, label, cv::Point(px - 10, py - 10), font_face, font_scale, color, thickness);
        }
    }

    return image;
}

