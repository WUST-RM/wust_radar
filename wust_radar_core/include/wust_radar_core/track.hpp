#pragma once

#include <Eigen/Dense>
#include <builtin_interfaces/msg/detail/time__struct.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "wust_radar_core/kf.hpp"
#include "wust_radar_core/type/type.hpp"

enum class TrackStateEnum { INACTIVE, TENTATIVE, CONFIRMED, LOST };

// 单次检测结果（检测输入）
struct Detection {
    int bot_id = 0;
    Box box;
    Eigen::Vector3d position; // 3D位置
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    float confidence = 0.f;
    float iou = 0.f;
};
inline std::vector<Detection> rawCarsToDetections(const RawCars& raw_cars_msg) {
    std::vector<Detection> detections;

    for (const auto& raw_car: raw_cars_msg.raw_cars) {
        Detection det;

        det.bot_id = static_cast<int>(raw_car.car_class);
        det.box = raw_car.box;

        // 3D位置直接用uwb_point
        det.position.x() = raw_car.uwb_point.x();
        det.position.y() = raw_car.uwb_point.y();
        det.position.z() = 0.0;

        det.velocity = Eigen::Vector3d::Zero(); // 速度一般初始化为0，如果有历史可估计

        det.confidence = 0.f;
        // confidence可以取armors里最大置信度，或者自己定义：
        float max_conf = 0.f;
        for (const auto& armor: raw_car.armors) {
            if (armor.confidence > max_conf) {
                max_conf = armor.confidence;
            }
        }
        det.confidence = max_conf;

        // iou初始化为0
        det.iou = 0.f;

        detections.push_back(det);
    }

    return detections;
}

// 轨迹类
class Track {
public:
    int track_id;
    int bot_id = 0;
    std::deque<int> bot_id_history_;
    TrackStateEnum state = TrackStateEnum::INACTIVE;

    int hit_count = 0;
    int miss_count = 0;

    Box box; // 当前边界框
    Eigen::Vector3d position; // 估计的3D位置
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    float confidence = 0.f;
    TrackCfg cfg;
    // 卡尔曼滤波状态
    Eigen::VectorXd pos_measurement;
    Eigen::VectorXd pos_kf_state;
    std::shared_ptr<KalmanFilter3D> pos_kf;
    Eigen::VectorXd box_measurement;
    Eigen::VectorXd box_kf_state;
    std::shared_ptr<KalmanFilterBox> box_kf;
    double last_ypd_y = 0;
    double dt;
    std::chrono::steady_clock::time_point last_update_time;

public:
    Track() = default;
    Track(int id, const Detection& det, const TrackCfg& cfg);

    void predict(double dt);
    void update(const Detection& det);

    Eigen::Vector3d getPredictedPosition() const;

    // 更新边界框
    void updateBox(const Box& new_box);
};
inline RightCars
tracksToRightCars(const std::vector<Track>& tracks, const builtin_interfaces::msg::Time& ros_time) {
    RightCars right_cars_msg;
    right_cars_msg.ros_time = ros_time;

    for (const auto& track: tracks) {
        // 只转换活跃或确认的轨迹（根据你实际需要）
        if (track.state == TrackStateEnum::CONFIRMED) {
            RightCar rc;
            rc.car_class = static_cast<CarClass>(track.bot_id); // 转回CarClass枚举
            rc.uwb_point = track.position;
            right_cars_msg.right_cars.push_back(rc);
        }
    }

    return right_cars_msg;
}
