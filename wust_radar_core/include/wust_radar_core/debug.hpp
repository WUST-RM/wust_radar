#pragma once
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "wust_radar_core/type/type.hpp"
cv::Mat DrawPointsOnImage(const cv::Mat& image_in, const FinalCars& cars);