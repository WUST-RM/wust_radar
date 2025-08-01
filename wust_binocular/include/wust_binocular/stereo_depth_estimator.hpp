#pragma once
#include "wust_binocular/type/type.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
//#include <opencv2/cudastereo.hpp>
#include <iostream>

class StereoDepthEstimator {
public:
    StereoDepthEstimator(
        const cv::Mat& K1,
        const cv::Mat& D1,
        const cv::Mat& K2,
        const cv::Mat& D2,
        const cv::Mat& R,
        const cv::Mat& T,
        bool prefer_cuda = false // 是否优先用cuda
    ):
        K1_(K1.clone()),
        D1_(D1.clone()),
        K2_(K2.clone()),
        D2_(D2.clone()),
        R_(R.clone()),
        T_(T.clone()),
        prefer_cuda_(prefer_cuda),
        is_ready_(false),
        use_cuda_(false) {
        use_cuda_ = prefer_cuda_ && (cv::cuda::getCudaEnabledDeviceCount() > 0);
        if (use_cuda_) {
            std::cout << "[StereoDepthEstimator] CUDA device detected. Using CUDA acceleration.\n";
        } else {
            std::cout << "[StereoDepthEstimator] CUDA not available or disabled. Using CPU.\n";
        }
    }

    cv::Mat computeDepth(const CommonFrame& frame) {
        if (!is_ready_) {
            if (frame.image_L.empty()) {
                throw std::runtime_error(
                    "Left image is empty, cannot initialize StereoDepthEstimator."
                );
            }
            init(frame.image_L.size());
        }

        // 1. 校正（remap）
        cv::Mat rect_L, rect_R;
        cv::remap(frame.image_L, rect_L, mapLx_, mapLy_, cv::INTER_LINEAR);
        cv::remap(frame.image_R, rect_R, mapRx_, mapRy_, cv::INTER_LINEAR);

        if (use_cuda_) {
            return computeDepthCuda(rect_L, rect_R);
        } else {
            return computeDepthCpu(rect_L, rect_R);
        }
    }
    cv::Mat computeDepth(const CommonFrame& frame, const cv::Rect& roi) {
        if (!is_ready_) {
            if (frame.image_L.empty()) {
                throw std::runtime_error(
                    "Left image is empty, cannot initialize StereoDepthEstimator."
                );
            }
            init(frame.image_L.size());
        }

        // 先对整个图像做 remap，再裁剪 ROI
        cv::Mat rect_L_full, rect_R_full;
        cv::remap(frame.image_L, rect_L_full, mapLx_, mapLy_, cv::INTER_LINEAR);
        cv::remap(frame.image_R, rect_R_full, mapRx_, mapRy_, cv::INTER_LINEAR);

        // 裁剪ROI（防止越界）
        cv::Rect bounded_roi = roi & cv::Rect(0, 0, rect_L_full.cols, rect_L_full.rows);

        cv::Mat rect_L = rect_L_full(bounded_roi);
        cv::Mat rect_R = rect_R_full(bounded_roi);

        if (use_cuda_) {
            return computeDepthCuda(rect_L, rect_R);
        } else {
            return computeDepthCpu(rect_L, rect_R);
        }
    }

private:
    void init(const cv::Size& image_size) {
        cv::Mat R1, R2, P1, P2;
        cv::stereoRectify(
            K1_,
            D1_,
            K2_,
            D2_,
            image_size,
            R_,
            T_,
            R1,
            R2,
            P1,
            P2,
            Q_,
            cv::CALIB_ZERO_DISPARITY,
            0,
            image_size
        );

        cv::initUndistortRectifyMap(K1_, D1_, R1, P1, image_size, CV_32FC1, mapLx_, mapLy_);
        cv::initUndistortRectifyMap(K2_, D2_, R2, P2, image_size, CV_32FC1, mapRx_, mapRy_);

        fx_ = P1.at<double>(0, 0);
        baseline_ = cv::norm(T_);

        if (use_cuda_) {
            // CUDA 版本：创建 StereoSGM，参数可以根据需要调整
            //stereo_cuda_ = cv::cuda::createStereoSGM(0, 64, 16);
        } else {
            // CPU 版本：创建 StereoSGBM，参数可调
            stereo_cpu_ = cv::StereoSGBM::create(0, 64, 5);
            stereo_cpu_->setP1(8 * 3 * 5 * 5);
            stereo_cpu_->setP2(32 * 3 * 5 * 5);
            stereo_cpu_->setPreFilterCap(31);
            stereo_cpu_->setMode(cv::StereoSGBM::MODE_SGBM);
        }

        is_ready_ = true;
    }

    cv::Mat computeDepthCpu(const cv::Mat& rect_L, const cv::Mat& rect_R) {
        cv::Mat gray_L, gray_R;
        cv::cvtColor(rect_L, gray_L, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rect_R, gray_R, cv::COLOR_BGR2GRAY);

        cv::Mat disparity_s16;
        stereo_cpu_->compute(gray_L, gray_R, disparity_s16);

        cv::Mat disparity_f32;
        disparity_s16.convertTo(disparity_f32, CV_32F, 1.0 / 16.0);

        return fx_ * baseline_ / (disparity_f32);
    }

    cv::Mat computeDepthCuda(const cv::Mat& rect_L, const cv::Mat& rect_R) {
        cv::Mat gray_L, gray_R;
        cv::cvtColor(rect_L, gray_L, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rect_R, gray_R, cv::COLOR_BGR2GRAY);

        cv::cuda::GpuMat d_left(gray_L);
        cv::cuda::GpuMat d_right(gray_R);
        cv::cuda::GpuMat d_disp(rect_L.size(), CV_16S);

        //stereo_cuda_->compute(d_left, d_right, d_disp);

        cv::Mat disparity_s16;
        d_disp.download(disparity_s16);

        cv::Mat disparity_f32;
        disparity_s16.convertTo(disparity_f32, CV_32F, 1.0 / 16.0);

        return fx_ * baseline_ / (disparity_f32 + 1e-6);
    }

private:
    // 内参和外参
    cv::Mat K1_, D1_, K2_, D2_, R_, T_;

    // rectification maps
    cv::Mat mapLx_, mapLy_, mapRx_, mapRy_;
    cv::Mat Q_;

    // stereo params
    double fx_ = 0;
    double baseline_ = 0;

    // 运行时控制
    bool prefer_cuda_;
    bool use_cuda_;
    bool is_ready_;

    // CPU 和 CUDA stereo 算法对象
    cv::Ptr<cv::StereoSGBM> stereo_cpu_;
    //cv::Ptr<cv::cuda::StereoSGM> stereo_cuda_;
};
