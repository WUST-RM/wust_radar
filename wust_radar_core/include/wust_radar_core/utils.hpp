#pragma once
#include "wust_radar_core/type/type.hpp"
#include "wust_radar_interfaces/msg/detect_result.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
namespace utils {
inline Eigen::Vector3d cvFrame2world(const Eigen::Vector3d& cv) {
    Eigen::Matrix3d R;
    R << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Vector3d world;
    world = R * cv;
    return world;
}
inline Eigen::Vector3d world2uwb(const Eigen::Vector3d& world, FACTION faction) {
    Eigen::Vector3d uwb;
    if (faction == FACTION::RAD) {
        uwb.x() = world.x() + 14.0;
        uwb.y() = world.y() + 7.5;
        uwb.z() = world.z();
    } else {
        uwb.x() = 28.0 - (world.x() + 14.0);
        uwb.y() = 14.0 - (world.y() + 7.5);
        uwb.z() = world.z();
    }

    return uwb;
}
inline bool isUnknown(CarClass car_class) {
    return car_class == CarClass::GUNKNOWN || car_class == CarClass::RUNKNOWN
        || car_class == CarClass::BUNKNOWN;
}
inline CarClass colorId2CarClass(int color, int id) {
    if (color == 0) {
        switch (id) {
            case 0:
                return CarClass::BUNKNOWN;
            case 1:
                return CarClass::B1;
            case 2:
                return CarClass::B2;
            case 3:
                return CarClass::B3;
            case 4:
                return CarClass::B4;
            case 6:
                return CarClass::B7;
            default:
                return CarClass::BUNKNOWN;
        }
    } else if (color == 2) {
        switch (id) {
            case 0:
                return CarClass::RUNKNOWN;
            case 1:
                return CarClass::R1;
            case 2:
                return CarClass::R2;
            case 3:
                return CarClass::R3;
            case 4:
                return CarClass::R4;
            case 6:
                return CarClass::R7;
            default:
                return CarClass::RUNKNOWN;
        }

    } else {
        switch (id) {
            case 0:
                return CarClass::GUNKNOWN;
            case 1:
                return CarClass::G1;
            case 2:
                return CarClass::G2;
            case 3:
                return CarClass::G3;
            case 4:
                return CarClass::G4;
            case 6:
                return CarClass::G7;
            default:
                return CarClass::GUNKNOWN;
        }
    }
}
inline std::pair<int, int> carClass2ColorId(CarClass c) {
    switch (c) {
        // 蓝色
        case CarClass::BUNKNOWN:
            return { 0, 0 };
        case CarClass::B1:
            return { 0, 1 };
        case CarClass::B2:
            return { 0, 2 };
        case CarClass::B3:
            return { 0, 3 };
        case CarClass::B4:
            return { 0, 4 };
        case CarClass::B7:
            return { 0, 6 };

        // 红色
        case CarClass::RUNKNOWN:
            return { 2, 0 };
        case CarClass::R1:
            return { 2, 1 };
        case CarClass::R2:
            return { 2, 2 };
        case CarClass::R3:
            return { 2, 3 };
        case CarClass::R4:
            return { 2, 4 };
        case CarClass::R7:
            return { 2, 6 };

        case CarClass::GUNKNOWN:
            return { 1, 0 };
        case CarClass::G1:
            return { 1, 1 };
        case CarClass::G2:
            return { 1, 2 };
        case CarClass::G3:
            return { 1, 3 };
        case CarClass::G4:
            return { 1, 4 };
        case CarClass::G7:
            return { 1, 6 };

        default:
            return { -1, -1 }; // 无效
    }
}

inline Box msgBox2Box(const wust_radar_interfaces::msg::Box& msg_box) {
    Box box;
    box.bottom = msg_box.bottom;
    box.top = msg_box.top;
    box.left = msg_box.left;
    box.right = msg_box.right;
    return box;
}
inline cv::Point2d armorBox2Key(const Box& armor_box) {
    cv::Point2d key;
    key.x = (armor_box.right + armor_box.left) / 2.0;
    key.y = armor_box.bottom;
    return key;
}
inline cv::Point2d carBox2Key(const Box& armor_box, double ratio) {
    cv::Point2d key;
    key.x = (armor_box.right + armor_box.left) / 2.0;
    key.y = armor_box.bottom + (armor_box.top - armor_box.bottom) * ratio;
    return key;
}
inline Armor msgArmor2Armor(const wust_radar_interfaces::msg::Armor& msg_armor) {
    Armor armor;
    armor.box = msgBox2Box(msg_armor.box);
    armor.confidence = msg_armor.confidence;
    armor.car_class = colorId2CarClass(msg_armor.color, msg_armor.number);
    armor.key_point = armorBox2Key(armor.box);
    return armor;
}
inline RawCars msgCar2RawCar(const wust_radar_interfaces::msg::DetectResult& msg_cars) {
    RawCars cars;
    cars.ros_time = msg_cars.header.stamp;
    for (auto msg_car: msg_cars.cars) {
        RawCar raw_car;
        raw_car.box = msgBox2Box(msg_car.car_box);
        raw_car.fall_back_point = carBox2Key(raw_car.box, 0.15);
        int max_color = -1;
        int max_carclass = 0;
        float max_confidence = 0;
        double key_y;
        for (auto msg_armor: msg_car.armors) {
            Armor armor = msgArmor2Armor(msg_armor);
            key_y = armor.key_point.y + key_y;
            raw_car.armors.push_back(armor);
            if (armor.confidence > max_confidence) {
                max_confidence = armor.confidence;
                max_color = msg_armor.color;
                max_carclass = msg_armor.number;
            }
        }
        raw_car.key_point.y = key_y / raw_car.armors.size();
        raw_car.key_point.x = raw_car.fall_back_point.x;
        raw_car.car_class = colorId2CarClass(max_color, max_carclass);
        cars.raw_cars.push_back(raw_car);
    }
    return cars;
}
} // namespace utils