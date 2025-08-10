#include "wust_radar_core/pixel_to_world.hpp"
#include "wust_utils/logger.hpp"
PixelToWorld::PixelToWorld():
    camera_matrix_(cv::Mat(3, 3, CV_32F, cv::Scalar(0))),
    rotation_matrix_(cv::Mat(3, 3, CV_64F, cv::Scalar(0))),
    translation_vector_(cv::Mat(3, 1, CV_64F, cv::Scalar(0))) {}

bool PixelToWorld::LoadCameraParameters(const std::string& yaml_file) {
    YAML::Node config = YAML::LoadFile(yaml_file);

    auto camera_mat_node = config["camera_matrix"];
    if (!camera_mat_node) {
        WUST_ERROR("LoadCameraParameters") << "缺少camera_matrix";
        return false;
    }
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            camera_matrix_.at<float>(r, c) = camera_mat_node["data"][r * 3 + c].as<float>();
        }
    }

    auto rot_node = config["rotation_matrix"];
    if (!rot_node) {
        WUST_ERROR("LoadCameraParameters") << "缺少rotation_matrix";
        return false;
    }
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            rotation_matrix_.at<double>(r, c) = rot_node["data"][r * 3 + c].as<double>();
        }
    }

    auto trans_node = config["translation_vector"];
    if (!trans_node) {
        WUST_ERROR("LoadCameraParameters") << "缺少translation_vector";
        return false;
    }
    for (int r = 0; r < 3; r++) {
        translation_vector_.at<double>(r, 0) = trans_node["data"][r].as<double>();
    }

    // 相机外参矩阵 [R|t]
    extrinsic_ = cv::Mat::eye(4, 4, CV_64F);
    rotation_matrix_.copyTo(extrinsic_(cv::Rect(0, 0, 3, 3)));
    translation_vector_.copyTo(extrinsic_(cv::Rect(3, 0, 1, 3)));

    // 内参矩阵 3x3 float 转 double
    intrinsic_ = cv::Mat::eye(3, 3, CV_64F);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            intrinsic_.at<double>(r, c) = static_cast<double>(camera_matrix_.at<float>(r, c));
        }
    }

    // 相机位置（世界坐标系中的相机中心） = -R^T * t
    cv::Mat R = rotation_matrix_;
    cv::Mat t = translation_vector_;
    cv::Mat cam_center = -R.t() * t;
    camera_position_ = Eigen::Vector3d(
        cam_center.at<double>(0),
        cam_center.at<double>(1),
        cam_center.at<double>(2)
    );

    return true;
}

bool PixelToWorld::LoadPLY(const std::string& ply_file) {
    mesh_ = open3d::io::CreateMeshFromFile(ply_file);
    if (!mesh_) {
        WUST_ERROR("LoadCameraParameters") << "无法加载PLY文件: " << ply_file;
        return false;
    }

    // 创建Open3D场景用于射线查询
    scene_ = std::make_shared<open3d::t::geometry::TriangleMesh>(
        open3d::t::geometry::TriangleMesh::FromLegacy(*mesh_)
    );

    scene3d_ = std::make_shared<open3d::t::geometry::TriangleMesh>(scene_->Clone());

    raycasting_scene_ = std::make_shared<open3d::t::geometry::RaycastingScene>();
    raycasting_scene_->AddTriangles(*scene_);

    WUST_INFO("LoadPLY") << "PLY网格加载成功, 顶点数: " << mesh_->vertices_.size();
    return true;
}

std::optional<Eigen::Vector3d> PixelToWorld::PixelTo3DPoint(const cv::Point2d& pixel) {
    // 1. 计算归一化像素坐标
    double x_norm = (pixel.x - intrinsic_.at<double>(0, 2)) / intrinsic_.at<double>(0, 0);
    double y_norm = (pixel.y - intrinsic_.at<double>(1, 2)) / intrinsic_.at<double>(1, 1);

    Eigen::Vector3d ray_dir_cam(x_norm, y_norm, 1.0);
    ray_dir_cam.normalize();

    // 2. 相机坐标系方向转世界坐标系
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R(r, c) = rotation_matrix_.at<double>(r, c);
    Eigen::Vector3d ray_dir_world = R.transpose() * ray_dir_cam;
    Eigen::Vector3d ray_orig = camera_position_;

    // 3. 构造 (1, 6) Tensor: [ox, oy, oz, dx, dy, dz]
    std::vector<float> rays_vec = {
        static_cast<float>(ray_orig.x()),      static_cast<float>(ray_orig.y()),
        static_cast<float>(ray_orig.z()),      static_cast<float>(ray_dir_world.x()),
        static_cast<float>(ray_dir_world.y()), static_cast<float>(ray_dir_world.z())
    };
    open3d::core::Tensor rays_tensor(rays_vec, { 1, 6 }, open3d::core::Dtype::Float32);

    // 4. 调用 CastRays (只传入 rays_tensor)
    auto result = raycasting_scene_->CastRays(rays_tensor);

    // 5. 取出 t_hit
    auto t_hit = result["t_hit"];
    if (t_hit.NumElements() == 0) {
        return std::nullopt;
    }

    // 把 t_hit 转成 std::vector<float>
    auto t_hit_vec = t_hit.ToFlatVector<float>();

    if (t_hit_vec.empty() || t_hit_vec[0] < 0.0f) {
        return std::nullopt;
    }

    float t = t_hit_vec[0];

    // 7. 计算交点
    Eigen::Vector3d pt_intersect = ray_orig + t * ray_dir_world;
    return pt_intersect;
}
