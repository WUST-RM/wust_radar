#include "wust_radar_core/car_pool.hpp"
#include "wust_utils/timer.hpp"
CarPool::CarPool(TrackCfg track_cfg) {
    track_cfg_ = track_cfg;
    auto initClass = [](CarClass car_class, FinalCar& car) {
        car.car_class = car_class;
        car.timestamp = std::chrono::steady_clock::now();
        car.uwb_point = Eigen::Vector3d::Zero();
    };
    initClass(CarClass::R1, r1);
    initClass(CarClass::R2, r2);
    initClass(CarClass::R3, r3);
    initClass(CarClass::R4, r4);
    initClass(CarClass::R7, r7);
    initClass(CarClass::B1, b1);
    initClass(CarClass::B2, b2);
    initClass(CarClass::B3, b3);
    initClass(CarClass::B4, b4);
    initClass(CarClass::B7, b7);
    point_guesser_ = std::make_unique<PointGuesser>(track_cfg_.guess_pts_path);
}

FinalCars CarPool::update(const TrackedCars& cars, FACTION faction) {
    faction_ = faction;
    std::vector<const TrackedCar*> matched;

    auto record_match = [&](CarClass cc, FinalCar& fc) {
        auto found = search(cc, cars);
        const TrackedCar* best = match(found, fc);
        if (best)
            matched.push_back(best);
    };

    record_match(CarClass::R1, r1);
    record_match(CarClass::R2, r2);
    record_match(CarClass::R3, r3);
    record_match(CarClass::R4, r4);
    record_match(CarClass::R7, r7);
    record_match(CarClass::B1, b1);
    record_match(CarClass::B2, b2);
    record_match(CarClass::B3, b3);
    record_match(CarClass::B4, b4);
    record_match(CarClass::B7, b7);

    unknown_cars.clear();
    for (auto& c: cars.tracked_cars) {
        bool is_matched = false;
        for (auto ptr: matched) {
            if (&c == ptr) {
                is_matched = true;
                break;
            }
        }
        if (!is_matched) {
            unknown_cars.push_back(c);
        }
    }
    checkExpiredCars(1.0);
    std::vector<FinalCar*> need_guesses = { &r1, &r2, &r3, &r4, &r7, &b1, &b2, &b3, &b4, &b7 };
    for (auto* c: need_guesses) {
        if (c->state == CarState::NEEDGUESS) {
            c->uwb_point = point_guesser_->predict(*c);
            c->uwb_velocity = Eigen::Vector3d::Zero();
            c->state = CarState::GUESSING;
        }
    }
    return getAllCars();
}

std::vector<TrackedCar> CarPool::search(CarClass car_class, const TrackedCars& tracks) {
    std::vector<TrackedCar> ret;
    for (auto& track: tracks.tracked_cars) {
        if (track.car_class == car_class) {
            ret.push_back(track);
        }
    }
    return ret;
}
const TrackedCar* CarPool::match(const std::vector<TrackedCar>& tracks, FinalCar& car) {
    const TrackedCar* best_car = nullptr;
    size_t best_sixz = 0;

    for (const auto& track: tracks) {
        if (track.frame_count > best_sixz) {
            best_car = &track;
            best_sixz = track.frame_count;
        }
    }

    if (best_car) {
        car.uwb_point = best_car->uwb_point;
        car.uwb_velocity = best_car->uwb_velocity;
        car.timestamp = time_utils::now();
        car.state = CarState::ACTIVE;
        car.frame_count++;
    }

    return best_car;
}

void CarPool::checkExpiredCars(double time_thresh) {
    using namespace std::chrono;
    auto now = time_utils::now(); // steady_clock::time_point

    auto check = [&](FinalCar& car) {
        auto diff = duration_cast<seconds>(now - car.timestamp).count();
        if (diff > time_thresh) {
            car.state = CarState::NEEDGUESS;
        }
    };

    check(r1);
    check(r2);
    check(r3);
    check(r4);
    check(r7);
    check(b1);
    check(b2);
    check(b3);
    check(b4);
    check(b7);

    return;
}
FinalCars CarPool::getAllCars() {
    FinalCars fc;
    fc.final_cars.push_back(r1);
    fc.final_cars.push_back(r2);
    fc.final_cars.push_back(r3);
    fc.final_cars.push_back(r4);
    fc.final_cars.push_back(r7);
    fc.final_cars.push_back(b1);
    fc.final_cars.push_back(b2);
    fc.final_cars.push_back(b3);
    fc.final_cars.push_back(b4);
    fc.final_cars.push_back(b7);
    fc.timestamp = time_utils::now();
    return fc;
}