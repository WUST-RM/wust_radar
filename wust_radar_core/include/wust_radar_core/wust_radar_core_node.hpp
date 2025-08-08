#pragma once
#include "rclcpp/rclcpp.hpp"
#include "wust_radar_core/cascade_match_tracker.hpp"
#include "wust_radar_core/pixel_to_world.hpp"
#include "wust_radar_core/type/type.hpp"
#include "wust_radar_interfaces/msg/detect_result.hpp"
class WustRadarCoreNode: public rclcpp::Node {
public:
    WustRadarCoreNode(const rclcpp::NodeOptions& options);
    ~WustRadarCoreNode();
    void detectCallback(const wust_radar_interfaces::msg::DetectResult::SharedPtr msg);

private:
    rclcpp::Subscription<wust_radar_interfaces::msg::DetectResult>::SharedPtr detect_sub_;
    std::unique_ptr<PixelToWorld> pixel_to_world_;
    cv::Mat map_;
    std::unique_ptr<CascadeMatchTracker> tracker_;
};