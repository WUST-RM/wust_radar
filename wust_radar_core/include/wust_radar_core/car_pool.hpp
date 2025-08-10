#pragma once
#include "wust_radar_core/type/type.hpp"
#include <yaml-cpp/yaml.h>
class PointGuesser {
public:
    PointGuesser(const std::string& config_path) {
        YAML::Node config = YAML::LoadFile(config_path);
        cos_factor_ = config["cos_factor"] ? config["cos_factor"].as<double>() : 0.003;
        d_factor_ = config["d_factor"] ? config["d_factor"].as<double>() : 0.1;

        if (config["guess_points"]) {
            for (auto it = config["guess_points"].begin(); it != config["guess_points"].end(); ++it)
            {
                std::string name = it->first.as<std::string>();
                for (const auto& pt: it->second) {
                    double x = pt[0].as<double>();
                    double y = pt[1].as<double>();
                    double z = 0.0;
                    guess_points_[name].emplace_back(x, y, z);
                }
            }
        }
    }

    Eigen::Vector3d predict(const FinalCar& car) const {
        std::string name = carName(car.car_class);
        auto points = get_guess_points_for_robot(name);

        Eigen::Vector3d last_pos = car.uwb_point;

        if (last_pos.isZero(1e-4) || car.frame_count < 1) {
            if (!points.empty()) {
                return points.front();
            } else {
                return Eigen::Vector3d::Zero();
            }
        }

        Eigen::Vector3d v_vec = car.uwb_velocity;

        double best_score = -1e9;
        Eigen::Vector3d best_point;

        for (const auto& point: points) {
            Eigen::Vector3d d_vector = point - last_pos;
            double dot_product = v_vec.dot(d_vector);
            double v_norm = v_vec.norm();
            double d_norm = d_vector.norm();
            double cos_sim = dot_product / (v_norm * d_norm + 1e-8);
            double d_score = std::exp(-d_norm * d_factor_);
            double score = cos_factor_ * cos_sim + (1 - cos_factor_) * d_score;
            if (score > best_score) {
                best_score = score;
                best_point = point;
            }
        }
        return best_point;
    }

private:
    double cos_factor_;
    double d_factor_;
    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> guess_points_;

    std::vector<Eigen::Vector3d> get_guess_points_for_robot(const std::string& name) const {
        auto it = guess_points_.find(name);
        if (it == guess_points_.end())
            return {};

        return it->second;
    }

    std::string carName(CarClass cc) const {
        switch (cc) {
            case CarClass::R1:
                return "R1";
            case CarClass::R2:
                return "R2";
            case CarClass::R3:
                return "R3";
            case CarClass::R4:
                return "R4";
            case CarClass::R7:
                return "R7";
            case CarClass::B1:
                return "B1";
            case CarClass::B2:
                return "B2";
            case CarClass::B3:
                return "B3";
            case CarClass::B4:
                return "B4";
            case CarClass::B7:
                return "B7";
            default:
                return "";
        }
    }
};

class CarPool {
public:
    CarPool(TrackCfg track_cfg);
    FinalCars update(const TrackedCars& cars, FACTION faction);
    std::vector<TrackedCar> search(CarClass car_class, const TrackedCars& tracks);
    const TrackedCar* match(const std::vector<TrackedCar>& tracks, FinalCar& car);
    void checkExpiredCars(double time_thresh);
    FinalCars getAllCars();
    FinalCar r1;
    FinalCar r2;
    FinalCar r3;
    FinalCar r4;
    FinalCar r7;
    FinalCar b1;
    FinalCar b2;
    FinalCar b3;
    FinalCar b4;
    FinalCar b7;
    std::vector<TrackedCar> unknown_cars;
    FACTION faction_;
    std::unique_ptr<PointGuesser> point_guesser_;
    TrackCfg track_cfg_;
};