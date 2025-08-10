#include "wust_radar_core/wust_radar_core_node.hpp"
#include "wust_radar_core/debug.hpp"
#include "wust_radar_core/utils.hpp"
#include "wust_utils/logger.hpp"
#include "wust_utils/timer.hpp"
WustRadarCoreNode::WustRadarCoreNode(const rclcpp::NodeOptions& options):
    Node("wust_radar_core_node", options) {
    WUST_WARN("记住了") << "还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！";
    pixel_to_world_ = std::make_unique<PixelToWorld>();
    this->declare_parameter<int>("common.faction", 0);
    this->declare_parameter<std::string>("common.camera_info_path", "");
    this->declare_parameter<std::string>("common.ply_file", "");
    this->declare_parameter<std::string>("common.field_image", "");
    pixel_to_world_->LoadCameraParameters(this->get_parameter("common.camera_info_path").as_string()
    );
    pixel_to_world_->LoadPLY(this->get_parameter("common.ply_file").as_string());
    map_ = cv::imread(this->get_parameter("common.field_image").as_string());
    if (map_.empty()) {
        std::cerr << "无法加载图片\n";
    }
    faction_ = static_cast<FACTION>(this->get_parameter("common.faction").as_int());
    WUST_MAIN("main") << FactionToString(faction_);
    TrackCfg track_cfg;
    this->declare_parameter("track.max_match_m", 2.0);
    this->declare_parameter("track.max_miss_time", 1.0);
    this->declare_parameter("track.v_damping", 0.5);
    this->declare_parameter("track.w_dist", 0.5);
    this->declare_parameter("track.w_iou", 0.5);
    this->declare_parameter("track.w_botid", 0.5);
    this->declare_parameter("track.w_speed", 0.5);
    this->declare_parameter("track.track_theresh", 5);
    this->declare_parameter("track.guess_pts_path", "");
    double max_match_m = this->get_parameter("track.max_match_m").as_double();
    double max_miss_time = this->get_parameter("track.max_miss_time").as_double();
    track_cfg.v_damping = this->get_parameter("track.v_damping").as_double();
    track_cfg.w_dist = this->get_parameter("track.w_dist").as_double();
    track_cfg.w_iou = this->get_parameter("track.w_iou").as_double();
    track_cfg.w_botid = this->get_parameter("track.w_botid").as_double();
    track_cfg.w_speed = this->get_parameter("track.w_speed").as_double();
    track_cfg.track_theresh = this->get_parameter("track.track_theresh").as_int();
    track_cfg.guess_pts_path = this->get_parameter("track.guess_pts_path").as_string();
    tracker_ = std::make_unique<CascadeMatchTracker>(track_cfg, max_match_m, max_miss_time);
    car_pool_ = std::make_unique<CarPool>(track_cfg);
    detect_sub_ = this->create_subscription<wust_radar_interfaces::msg::DetectResult>(
        "detect_result",
        rclcpp::SensorDataQoS(),
        std::bind(&WustRadarCoreNode::detectCallback, this, std::placeholders::_1)
    );
}
WustRadarCoreNode::~WustRadarCoreNode() {}
//还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
//还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
//还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！//还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！//还不清楚裁判系统坐标，暂定！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

void WustRadarCoreNode::detectCallback(const wust_radar_interfaces::msg::DetectResult::SharedPtr msg
) {
    auto raw_cars = utils::msgCar2RawCar(*msg);
    for (auto& car: raw_cars.raw_cars) {
        auto point = pixel_to_world_->PixelTo3DPoint(car.fall_back_point);
        if (point) {
            auto world_point = utils::cvFrame2world(*point);
            auto uwb_point = utils::world2uwb(world_point, faction_);
            car.uwb_point = uwb_point;
        }
    }

    tracker_->update(rawCarsToDetections(raw_cars));
    auto tracked_cars = tracksToTrackedCars(tracker_->getTracks(), raw_cars.ros_time);
    auto final_cars = car_pool_->update(tracked_cars, faction_);
    if (!map_.empty()) {
        cv::Mat image = DrawPointsOnImage(map_, final_cars);
        cv::imshow("map", image);
        cv::waitKey(1);
    }
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(WustRadarCoreNode)