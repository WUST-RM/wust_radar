#pragma once
#include <iostream>
#include <open3d/Open3D.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <yaml-cpp/yaml.h>

class PixelToWorld {
public:
    PixelToWorld();

    bool LoadCameraParameters(const std::string& yaml_file);

    bool LoadPLY(const std::string& ply_file);

    /// 输入像素点，返回射线与模型交点（世界坐标）
    /// 如果没有交点，返回空optional
    std::optional<Eigen::Vector3d> PixelTo3DPoint(const cv::Point2d& pixel);

private:
    cv::Mat camera_matrix_;
    cv::Mat rotation_matrix_;
    cv::Mat translation_vector_;
    cv::Mat extrinsic_;
    cv::Mat intrinsic_;

    Eigen::Vector3d camera_position_; // 相机中心世界坐标

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_;

    // 新的Open3D Tensor版本Mesh和场景
    std::shared_ptr<open3d::t::geometry::TriangleMesh> scene_;
    std::shared_ptr<open3d::t::geometry::TriangleMesh> scene3d_;
    std::shared_ptr<open3d::t::geometry::RaycastingScene> raycasting_scene_;
};
