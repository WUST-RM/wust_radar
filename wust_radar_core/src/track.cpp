
#include "wust_radar_core/track.hpp"
#include "wust_radar_core/utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

Track::Track(int id, const Detection& det, const TrackCfg& cfg):
    track_id(id),
    bot_id(det.bot_id),
    state(TrackStateEnum::TENTATIVE),
    hit_count(1),
    miss_count(0),
    box(det.box),
    position(det.position),
    velocity(det.velocity),
    confidence(det.confidence),
    cfg(cfg),
    pos_measurement(Eigen::VectorXd::Zero(3)),
    pos_kf_state(Eigen::VectorXd::Zero(6)),
    box_measurement(Eigen::VectorXd::Zero(4)),
    box_kf_state(Eigen::VectorXd::Zero(8)) {
    pos_kf = std::make_shared<KalmanFilter3D>(det.position, 0.05);
    box_kf = std::make_shared<KalmanFilterBox>(
        Eigen::Vector4d(det.box.centerx(), det.box.centery(), det.box.width(), det.box.centery()),
        0.05
    );
    last_update_time = std::chrono::steady_clock::now();
}

void Track::predict(double dt) {
    this->dt = dt;
    pos_kf->setDt(dt);
    pos_kf_state = pos_kf->predict();
    position.x() = pos_kf_state(0);
    position.y() = pos_kf_state(1);
    position.z() = pos_kf_state(2);
    velocity.x() = pos_kf_state(3);
    velocity.y() = pos_kf_state(4);
    velocity.z() = pos_kf_state(5);
    box_kf->setDt(dt);
    box_kf_state = box_kf->predict();
    double center_x = box_kf_state[0];
    double center_y = box_kf_state[1];
    double width = box_kf_state[2];
    double height = box_kf_state[3];

    Box box;
    box.left = center_x - width / 2.0f;
    box.right = center_x + width / 2.0f;
    box.top = center_y - height / 2.0f;
    box.bottom = center_y + height / 2.0f;
}

void Track::update(const Detection& det) {
    auto p = det.position;
    pos_measurement = Eigen::Vector3d(p.x(), p.y(), p.z());
    pos_kf->update(pos_measurement);
    box_measurement = Eigen::Vector4d(box.centerx(), box.centery(), box.width(), box.height());
    confidence = det.confidence;
    last_update_time = std::chrono::steady_clock::now();

    hit_count++;
    miss_count = 0;
    if (!utils::isUnknown(static_cast<CarClass>(det.bot_id))) {
        bot_id_history_.push_back(det.bot_id);
    }

    if (bot_id_history_.size() > 100) {
        bot_id_history_.pop_front();
    }

    std::unordered_map<int, int> freq;
    for (int id: bot_id_history_) {
        freq[id]++;
    }

    int max_count = 0;
    int most_common_bot_id = -1;
    for (const auto& kv: freq) {
        if (kv.second > max_count) {
            max_count = kv.second;
            most_common_bot_id = kv.first;
        }
    }
    bot_id = most_common_bot_id;
}

Eigen::Vector3d Track::getPredictedPosition() const {
    return position;
}

void Track::updateBox(const Box& new_box) {
    box = new_box;
}
