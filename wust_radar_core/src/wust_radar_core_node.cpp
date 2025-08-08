#include "wust_radar_core/wust_radar_core_node.hpp"
#include "wust_radar_core/debug.hpp"
#include "wust_radar_core/utils.hpp"
#include "wust_utils/timer.hpp"
WustRadarCoreNode::WustRadarCoreNode(const rclcpp::NodeOptions& options):
    Node("wust_radar_core_node", options) {
    pixel_to_world_ = std::make_unique<PixelToWorld>();
    this->declare_parameter<std::string>("common.camera_info_path", "");
    this->declare_parameter<std::string>("common.ply_file", "");
    pixel_to_world_->LoadCameraParameters(this->get_parameter("common.camera_info_path").as_string()
    );
    pixel_to_world_->LoadPLY(this->get_parameter("common.ply_file").as_string());
    map_ = cv::imread("/home/hy/wust_radar/src/wust_radar_core/field_image.png");
    if (map_.empty()) {
        std::cerr << "无法加载图片\n";
    }
    TrackCfg track_cfg;
    this->declare_parameter("track.max_match_m", 2.0);
    this->declare_parameter("track.max_miss_time", 1.0);
    this->declare_parameter("track.v_damping", 0.5);
    this->declare_parameter("track.w_dist", 0.5);
    this->declare_parameter("track.w_iou", 0.5);
    this->declare_parameter("track.w_botid", 0.5);
    this->declare_parameter("track.w_speed", 0.5);
    this->declare_parameter("track.track_theresh", 5);
    double max_match_m = this->get_parameter("track.max_match_m").as_double();
    double max_miss_time = this->get_parameter("track.max_miss_time").as_double();
    track_cfg.v_damping = this->get_parameter("track.v_damping").as_double();
    track_cfg.w_dist = this->get_parameter("track.w_dist").as_double();
    track_cfg.w_iou = this->get_parameter("track.w_iou").as_double();
    track_cfg.w_botid = this->get_parameter("track.w_botid").as_double();
    track_cfg.w_speed = this->get_parameter("track.w_speed").as_double();
    track_cfg.track_theresh = this->get_parameter("track.track_theresh").as_int();
    tracker_ = std::make_unique<CascadeMatchTracker>(track_cfg, max_match_m, max_miss_time);
    detect_sub_ = this->create_subscription<wust_radar_interfaces::msg::DetectResult>(
        "detect_result",
        rclcpp::SensorDataQoS(),
        std::bind(&WustRadarCoreNode::detectCallback, this, std::placeholders::_1)
    );
}
WustRadarCoreNode::~WustRadarCoreNode() {}

void WustRadarCoreNode::detectCallback(const wust_radar_interfaces::msg::DetectResult::SharedPtr msg
) { 
    auto t1 =time_utils::now();
    auto raw_cars = utils::msgCar2RawCar(*msg);
    std::vector<Eigen::Vector3d> all_uwb_points;
    for (auto& car: raw_cars.raw_cars) {
        auto point = pixel_to_world_->PixelTo3DPoint(car.fall_back_point);
        if (point) {
            auto world_point = utils::cvFrame2world(*point);
            auto uwb_point = utils::world2uwb(world_point);
            car.uwb_point = uwb_point;
            // all_uwb_points.push_back(uwb_point);
        }
    }
    auto t2 = time_utils::now();
    tracker_->update(rawCarsToDetections(raw_cars));
    auto right_cars = tracksToRightCars(tracker_->getTracks(), raw_cars.ros_time);
    auto t3 = time_utils::now();
    //std::cout << "cost time: " << time_utils::durationMs(t2 ,t1) <<"  "<< time_utils::durationMs(t3 ,t2) <<"  "<< time_utils::durationMs(t3 ,t1) << std::endl;
    if (!map_.empty()) {
        cv::Mat image = DrawPointsOnImage(map_, right_cars);
        cv::imshow("map", image);
        cv::waitKey(1);
    }
}

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(WustRadarCoreNode)