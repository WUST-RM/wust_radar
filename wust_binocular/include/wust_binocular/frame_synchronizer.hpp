#pragma once
#include "wust_binocular/type/type.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>

using MatchFrameCallback = std::function<void(const FrameUnmatched&, const FrameUnmatched&)>;

class FrameSynchronizer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using FrameDur = Clock::duration;
    using FrameMap = std::map<TimePoint, FrameUnmatched>;

    FrameSynchronizer(
        double max_time_diff_s,
        double delay_s = 0.005,
        double expire_s = 1.0,
        bool use_id_match = false
    ):
        max_dt_(std::chrono::duration_cast<FrameDur>(std::chrono::duration<double>(max_time_diff_s))
        ),
        delay_(std::chrono::duration_cast<FrameDur>(std::chrono::duration<double>(delay_s))),
        expire_(std::chrono::duration_cast<FrameDur>(std::chrono::duration<double>(expire_s))),
        use_id_match_(use_id_match) {}

    void setMatchCallback(MatchFrameCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        match_cb_ = std::move(cb);
    }

    void pushFrame(const FrameUnmatched& frame) {
        TimePoint ts = frame.timestamp;
        std::lock_guard<std::mutex> lk(mtx_);

        auto& self_map = (frame.source == FrameSource::LEFT ? left_map_ : right_map_);
        auto& other_map = (frame.source == FrameSource::LEFT ? right_map_ : left_map_);

        // 1) 插入自己的有序容器，并记下迭代器
        auto self_it = self_map.emplace(ts, frame).first;

        if (use_id_match_) {
            // 2a) 精确 seq_id 匹配
            auto it = std::find_if(other_map.begin(), other_map.end(), [&](auto& p) {
                return p.second.seq_id == frame.seq_id;
            });
            if (it != other_map.end()) {
                dispatchMatch(frame, it->second);
                other_map.erase(it);
                self_map.erase(self_it); // <—— 同步清理本端帧
            }
        } else {
            // 2b) 时间戳最近邻匹配（带延迟窗口）
            TimePoint cutoff = ts - delay_;
            auto it_hi = other_map.lower_bound(cutoff);

            // 候选：it_hi 及其前一个
            std::vector<FrameMap::iterator> cands;
            if (it_hi != other_map.end())
                cands.push_back(it_hi);
            if (it_hi != other_map.begin())
                cands.push_back(std::prev(it_hi));

            // 选最小时间差
            FrameDur best_dt = max_dt_ + FrameDur(1);
            auto best_it = other_map.end();
            for (auto it: cands) {
                FrameDur dt = (ts > it->first ? ts - it->first : it->first - ts);
                if (dt < best_dt) {
                    best_dt = dt;
                    best_it = it;
                }
            }
            if (best_it != other_map.end() && best_dt <= max_dt_) {
                dispatchMatch(frame, best_it->second);
                other_map.erase(best_it);
                self_map.erase(self_it); // <—— 同步清理本端帧
            }
        }

        // 3) 批量清理过期帧
        cleanup(left_map_);
        cleanup(right_map_);
    }

private:
    void dispatchMatch(const FrameUnmatched& a, const FrameUnmatched& b) {
        if (!match_cb_)
            return;
        if (a.source == FrameSource::LEFT)
            match_cb_(a, b);
        else
            match_cb_(b, a);
    }

    void cleanup(FrameMap& mp) {
        TimePoint now = Clock::now();
        TimePoint cutoff = now - expire_;
        mp.erase(mp.begin(), mp.lower_bound(cutoff));
    }

    FrameMap left_map_, right_map_;
    FrameDur max_dt_, delay_, expire_;
    bool use_id_match_;
    MatchFrameCallback match_cb_;
    std::mutex mtx_;
};
