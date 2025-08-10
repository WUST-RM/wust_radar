#pragma once

#include <Eigen/Dense>
#include <builtin_interfaces/msg/detail/time__struct.hpp>
#include <opencv2/opencv.hpp>
enum class FACTION { RAD = 0, BULE = 1 };
inline std::string FactionToString(FACTION faction) {
    switch (faction) {
        case FACTION::RAD:
            return "RAD";
        case FACTION::BULE:
            return "BULE";
        default:
            return "UNKNOWN";
    }
}
enum class CarState { ACTIVE, NEEDGUESS, GUESSING };
enum class CarClass {
    RUNKNOWN = 10,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R7 = 7,
    BUNKNOWN = 110,
    B1 = 101,
    B2 = 102,
    B3 = 103,
    B4 = 104,
    B7 = 107,
    GUNKNOWN = -10,
    G1 = -1,
    G2 = -2,
    G3 = -3,
    G4 = -4,
    G7 = -7
};
struct Box {
    float left;
    float right;
    float top;
    float bottom;
    float width() const {
        return right - left;
    }
    float height() const {
        return bottom - top;
    }
    float centerx() const {
        return (left + right) / 2.0f;
    }
    float centery() const {
        return (top + bottom) / 2.0f;
    }
};
struct Armor {
    Box box;
    float confidence;
    CarClass car_class;
    cv::Point2d key_point;
};
struct RawCar {
    Box box;
    cv::Point2d fall_back_point;
    std::vector<Armor> armors;
    cv::Point2d key_point;
    CarClass car_class;
    Eigen::Vector3d uwb_point;
};
struct RawCars {
    std::vector<RawCar> raw_cars;
    builtin_interfaces::msg::Time ros_time;
};

struct TrackedCar {
    CarClass car_class;
    Eigen::Vector3d uwb_point;
    Eigen::Vector3d uwb_velocity;
    size_t frame_count = 0;
};
struct TrackedCars {
    std::vector<TrackedCar> tracked_cars;
    builtin_interfaces::msg::Time ros_time;
};
struct FinalCar {
    CarClass car_class;
    Eigen::Vector3d uwb_point;
    Eigen::Vector3d uwb_velocity;
    CarState state = CarState::NEEDGUESS;
    size_t frame_count = 0;
    std::chrono::steady_clock::time_point timestamp;
};
struct FinalCars {
    std::vector<FinalCar> final_cars;
    std::chrono::steady_clock::time_point timestamp;
};
struct TrackCfg {
    int track_theresh;
    double v_damping;
    double w_dist;
    double w_iou;
    double w_botid;
    double w_speed;
    std::string guess_pts_path;
};