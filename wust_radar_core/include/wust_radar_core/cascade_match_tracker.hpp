#pragma once

#include "wust_radar_core/track.hpp"
#include <Eigen/Dense>
#include <builtin_interfaces/msg/detail/time__struct.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
// 主跟踪器类
class CascadeMatchTracker {
public:
    CascadeMatchTracker(
        TrackCfg track_cfg,
        double max_match_distance = 2.0,
        double max_miss_time_ = 1.0
    );

    // 传入本帧检测结果，更新轨迹
    void update(const std::vector<Detection>& detections);

    // 获取当前轨迹列表
    std::vector<Track> getTracks();

private:
    std::mutex mtx_;

    std::unordered_map<int, Track> tracks_; // track_id -> Track
    int next_track_id_ = 0;

    double max_match_distance_;
    double max_miss_time_;
    int max_miss_count_;
    TrackCfg track_cfg_;

    // 计算匹配代价
    double computeCost(const Track& track, const Detection& det) const;

    // 使用匈牙利算法匹配轨迹和检测
    void matchTracksToDetections(
        const std::vector<Track*>& track_ptrs,
        const std::vector<const Detection*>& det_ptrs,
        std::vector<std::pair<int, int>>& matches,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_detections
    ) const;

    // 非极大抑制NMS
    void nonMaximumSuppression(std::vector<Detection>& detections, float iou_thresh = 0.5f);

    // 轨迹丢失点预测
    Eigen::Vector3d predictLostTrackPoint(const Track& track) const;

    // IOU计算函数
    float computeIOU(const Box& a, const Box& b) const;
};