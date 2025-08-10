#include "wust_radar_core/cascade_match_tracker.hpp"
#include "3rdparty/Hungarian.h"
#include "wust_radar_core/utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
// --------------------- CascadeMatchTracker --------------------------

CascadeMatchTracker::CascadeMatchTracker(
    TrackCfg track_cfg,
    double max_match_distance,
    double max_miss_time
):
    max_match_distance_(max_match_distance),
    max_miss_time_(max_miss_time),
    max_miss_count_(5),
    track_cfg_(track_cfg) {}

// IOU计算
float CascadeMatchTracker::computeIOU(const Box& a, const Box& b) const {
    float inter_left = std::max(a.left, b.left);
    float inter_right = std::min(a.right, b.right);
    float inter_top = std::max(a.top, b.top);
    float inter_bottom = std::min(a.bottom, b.bottom);

    float inter_width = inter_right - inter_left;
    float inter_height = inter_bottom - inter_top;
    if (inter_width <= 0 || inter_height <= 0)
        return 0.0f;

    float inter_area = inter_width * inter_height;
    float area_a = a.width() * a.height();
    float area_b = b.width() * b.height();

    return inter_area / (area_a + area_b - inter_area);
}

// NMS实现
void CascadeMatchTracker::nonMaximumSuppression(
    std::vector<Detection>& detections,
    float iou_thresh
) {
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i])
            continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j])
                continue;
            if (computeIOU(detections[i].box, detections[j].box) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<Detection> filtered;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i])
            filtered.push_back(detections[i]);
    }
    detections = std::move(filtered);
}

void CascadeMatchTracker::matchTracksToDetections(
    const std::vector<Track*>& tracks,
    const std::vector<const Detection*>& detections,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections
) const {
    if (tracks.empty() || detections.empty()) {
        // 无匹配，全部视为未匹配
        matches.clear();
        unmatched_tracks.clear();
        unmatched_detections.clear();

        for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
            unmatched_tracks.push_back(i);
        for (int j = 0; j < static_cast<int>(detections.size()); ++j)
            unmatched_detections.push_back(j);

        return;
    }
    int N = static_cast<int>(tracks.size());
    int M = static_cast<int>(detections.size());
    int L = std::max(N, M);

    // 构造方阵 cost 矩阵，初始化为大值
    std::vector<std::vector<double>> cost_matrix(L, std::vector<double>(L, 1e6));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double cost = computeCost(*tracks[i], *detections[j]);
            if (cost < 0)
                cost = 0; // 保证非负
            if (cost < max_match_distance_) {
                cost_matrix[i][j] = cost;
            }
        }
    }

    // 直接用 double 类型调用 Hungarian 算法，不必转整数矩阵
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    int cost_sum = HungAlgo.Solve(cost_matrix, assignment);

    matches.clear();
    unmatched_tracks.clear();
    unmatched_detections.clear();

    std::vector<bool> det_assigned(M, false);
    for (int i = 0; i < N; ++i) {
        int j = assignment[i];
        if (j >= 0 && j < M) {
            // 再次校验 cost 是否满足阈值，防止大值影响
            if (cost_matrix[i][j] < max_match_distance_) {
                matches.emplace_back(i, j);
                det_assigned[j] = true;
            } else {
                unmatched_tracks.push_back(i);
            }
        } else {
            unmatched_tracks.push_back(i);
        }
    }

    for (int j = 0; j < M; ++j) {
        if (!det_assigned[j])
            unmatched_detections.push_back(j);
    }
}

double CascadeMatchTracker::computeCost(const Track& track, const Detection& det) const {
    double dist = (track.getPredictedPosition() - det.position).norm();
    double iou = computeIOU(track.box, det.box);
    double bot_id_penalty = (track.bot_id != -1 && track.bot_id != det.bot_id) ? 10.0 : 0.0;
    double speed_diff = (track.velocity - det.velocity).norm();

    // 权重
    double W_DIST = track_cfg_.w_dist;
    double W_IOU = track_cfg_.w_iou;
    double W_BOTID = track_cfg_.w_botid;
    double W_SPEED = track_cfg_.w_speed;

    double cost = W_DIST * dist + W_IOU * iou + W_BOTID * bot_id_penalty + W_SPEED * speed_diff;
    return cost;
}

Eigen::Vector3d CascadeMatchTracker::predictLostTrackPoint(const Track& track) const {
    constexpr double dt = 0.1;
    return track.position + track.velocity * dt;
}

void CascadeMatchTracker::update(const std::vector<Detection>& detections) {
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<Detection> filtered_detections = detections;
    nonMaximumSuppression(filtered_detections);

    std::vector<Track*> track_ptrs;
    for (auto& kv: tracks_) {
        track_ptrs.push_back(&kv.second);
    }
    std::vector<const Detection*> det_ptrs;
    for (const auto& det: filtered_detections) {
        det_ptrs.push_back(&det);
    }

    auto now = std::chrono::steady_clock::now();
    double dt;
    for (auto* track: track_ptrs) {
        dt = std::chrono::duration<double>(now - track->last_update_time).count();
        track->predict(dt);
    }
    max_miss_count_ = max_miss_time_ / dt;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;

    matchTracksToDetections(track_ptrs, det_ptrs, matches, unmatched_tracks, unmatched_detections);

    // 更新匹配轨迹
    for (const auto& [t_idx, d_idx]: matches) {
        Track& track = *track_ptrs[t_idx];
        const Detection& det = *det_ptrs[d_idx];
        track.update(det);

        if (track.state == TrackStateEnum::INACTIVE) {
            track.state = TrackStateEnum::TENTATIVE;
            track.hit_count = 1;
            track.miss_count = 0;
        } else if (track.state == TrackStateEnum::TENTATIVE && track.hit_count >= track_cfg_.track_theresh)
        {
            track.state = TrackStateEnum::CONFIRMED;
        } else if (track.state == TrackStateEnum::LOST) {
            track.state = TrackStateEnum::CONFIRMED;
            track.miss_count = 0;
        }
    }

    // 处理未匹配轨迹
    for (int idx: unmatched_tracks) {
        Track& track = *track_ptrs[idx];
        track.miss_count++;
        Eigen::VectorXd pos_kf_state = track.pos_kf->getState();
        pos_kf_state(3) *= track_cfg_.v_damping;
        pos_kf_state(4) *= track_cfg_.v_damping;
        pos_kf_state(5) *= track_cfg_.v_damping;
        track.pos_kf->setState(pos_kf_state);
        Eigen::VectorXd box_kf_state = track.box_kf->getState();
        box_kf_state(4) *= track_cfg_.v_damping;
        box_kf_state(5) *= track_cfg_.v_damping;
        box_kf_state(6) *= track_cfg_.v_damping;
        box_kf_state(7) *= track_cfg_.v_damping;
        track.box_kf->setState(box_kf_state);
        if (track.miss_count > max_miss_count_) {
            track.position = predictLostTrackPoint(track);
            track.state = TrackStateEnum::LOST;
        }
    }

    // 新检测创建新轨迹
    for (int idx: unmatched_detections) {
        const Detection& det = *det_ptrs[idx];
        Track new_track(next_track_id_++, det, track_cfg_);
        tracks_[new_track.track_id] = new_track;
    }

    // 删除过期轨迹
    std::vector<int> to_erase;
    for (auto& [id, track]: tracks_) {
        if (track.state == TrackStateEnum::LOST && track.miss_count > 2 * max_miss_count_) {
            to_erase.push_back(id);
        }
    }
    for (int id: to_erase) {
        tracks_.erase(id);
    }
}

std::vector<Track> CascadeMatchTracker::getTracks() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<Track> result;
    for (auto& [id, track]: tracks_) {
        result.push_back(track);
    }
    return result;
}