#pragma once

#include "3rdparty/angles.h"
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
class KalmanFilter3D {
public:
    KalmanFilter3D(const Eigen::VectorXd& initial_pos = Eigen::VectorXd::Zero(3), double dt = 1.0):
        dt_(dt) {
        // 状态转移矩阵 F
        F_.setIdentity();
        F_(0, 3) = dt_;
        F_(1, 4) = dt_;
        F_(2, 5) = dt_;

        // 观测矩阵 H
        H_.setZero();
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;
        H_(2, 2) = 1.0;

        // 初始化状态向量 x
        x_.setZero();
        if (initial_pos.size() == 3) {
            x_.head<3>() = initial_pos;
        }

        // 初始化协方差矩阵 P
        P_.setIdentity();

        // 测量噪声协方差 R
        R_ = Eigen::Matrix3d::Identity() * 0.1;

        // 过程噪声协方差 Q
        Q_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
    }
    void setDt(double new_dt) {
        dt_ = new_dt;
        // 更新状态转移矩阵 F_ 里和 dt 相关的元素
        F_.setIdentity();
        F_(0, 3) = dt_;
        F_(1, 4) = dt_;
        F_(2, 5) = dt_;
    }
    Eigen::VectorXd getState() const {
        return x_;
    }

    void setState(const Eigen::VectorXd& new_x) {
        if (new_x.size() == x_.size()) {
            x_ = new_x;
        } else {
            std::cerr << "Error: new state vector size mismatch." << std::endl;
        }
    }
    // 预测函数，返回预测的状态向量（6维，位置+速度）
    Eigen::VectorXd predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        return x_;
    }

    // 更新函数，输入测量（位置3维），返回更新后的状态向量（6维）
    Eigen::VectorXd update(const Eigen::VectorXd& measurement) {
        if (measurement.size() != 3) {
            std::cerr << "Error: measurement vector size must be 3." << std::endl;
            return x_;
        }

        Eigen::Vector3d y = measurement - H_ * x_;
        Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
        Eigen::Matrix<double, 6, 3> K = P_ * H_.transpose() * S.inverse();

        x_ = x_ + K * y;
        Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
        P_ = (I - K * H_) * P_;

        return x_;
    }

    // 重置函数
    void reset(const Eigen::VectorXd& initial_pos = Eigen::VectorXd::Zero(3)) {
        x_.setZero();
        if (initial_pos.size() == 3) {
            x_.head<3>() = initial_pos;
        }
        P_.setIdentity();
        R_ = Eigen::Matrix3d::Identity() * 0.1;
        Q_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
    }

private:
    double dt_;
    Eigen::Matrix<double, 6, 1> x_; // 状态向量 (x,y,z,vx,vy,vz)
    Eigen::Matrix<double, 6, 6> P_; // 状态协方差
    Eigen::Matrix<double, 6, 6> Q_; // 过程噪声协方差
    Eigen::Matrix3d R_; // 测量噪声协方差
    Eigen::Matrix<double, 6, 6> F_; // 状态转移矩阵
    Eigen::Matrix<double, 3, 6> H_; // 观测矩阵
};
class KalmanFilterBox {
public:
    // 状态向量维度8： [center_x, center_y, width, height, vx, vy, vw, vh]
    // 观测向量维度4： [center_x, center_y, width, height]

    KalmanFilterBox(const Eigen::Vector4d& initial_bbox = Eigen::Vector4d::Zero(), double dt = 1.0):
        dt_(dt) {
        // 状态转移矩阵 F (8x8)
        F_.setIdentity();
        F_(0, 4) = dt_;
        F_(1, 5) = dt_;
        F_(2, 6) = dt_;
        F_(3, 7) = dt_;

        // 观测矩阵 H (4x8)
        H_.setZero();
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;
        H_(2, 2) = 1.0;
        H_(3, 3) = 1.0;

        // 初始化状态向量x_
        x_.setZero();
        x_.head<4>() = initial_bbox;

        // 初始化协方差矩阵P_
        P_.setIdentity();

        // 初始化过程噪声协方差Q_ (可根据需求调整)
        Q_ = Eigen::Matrix<double, 8, 8>::Identity() * 0.01;

        // 初始化测量噪声协方差R_ (4x4)
        R_ = Eigen::Matrix4d::Identity() * 0.1;
    }

    void setDt(double dt) {
        dt_ = dt;
        F_.setIdentity();
        F_(0, 4) = dt_;
        F_(1, 5) = dt_;
        F_(2, 6) = dt_;
        F_(3, 7) = dt_;
    }
    Eigen::VectorXd getState() const {
        return x_;
    }

    void setState(const Eigen::VectorXd& new_x) {
        if (new_x.size() == x_.size()) {
            x_ = new_x;
        } else {
            std::cerr << "Error: new state vector size mismatch." << std::endl;
        }
    }
    Eigen::Vector4d predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
        return x_.head<4>();
    }

    Eigen::Vector4d update(const Eigen::Vector4d& measurement) {
        // 跳变检测（类似Python版）
        if ((x_.head<2>() - measurement.head<2>()).cwiseAbs().maxCoeff() > 100.0) {
            reset(measurement);
            return x_.head<4>();
        }

        Eigen::Vector4d y = measurement - H_ * x_;
        Eigen::Matrix4d S = H_ * P_ * H_.transpose() + R_;
        Eigen::Matrix<double, 8, 4> K = P_ * H_.transpose() * S.inverse();

        x_ = x_ + K * y;
        Eigen::Matrix<double, 8, 8> I = Eigen::Matrix<double, 8, 8>::Identity();
        P_ = (I - K * H_) * P_;

        return x_.head<4>();
    }

    void reset(const Eigen::Vector4d& initial_bbox = Eigen::Vector4d::Zero()) {
        x_.setZero();
        x_.head<4>() = initial_bbox;
        P_.setIdentity();
    }

    // 获取当前状态，bbox和速度
    void getState(Eigen::Vector4d& bbox, Eigen::Vector4d& vel) const {
        bbox = x_.head<4>();
        vel = x_.tail<4>();
    }

    // bbox中心格式转角点格式 [x1,y1,x2,y2]
    Eigen::Vector4d getBBoxCorners() const {
        Eigen::Vector4d corners;
        double cx = x_(0), cy = x_(1), w = x_(2), h = x_(3);
        corners(0) = cx - w / 2.0;
        corners(1) = cy - h / 2.0;
        corners(2) = cx + w / 2.0;
        corners(3) = cy + h / 2.0;
        return corners;
    }

    // 从角点格式设置 bbox（直接更新状态）
    void setBBoxFromCorners(const Eigen::Vector4d& corners) {
        double cx = (corners(0) + corners(2)) / 2.0;
        double cy = (corners(1) + corners(3)) / 2.0;
        double w = corners(2) - corners(0);
        double h = corners(3) - corners(1);
        update(Eigen::Vector4d(cx, cy, w, h));
    }

private:
    double dt_;
    Eigen::Matrix<double, 8, 1> x_;
    Eigen::Matrix<double, 8, 8> P_;
    Eigen::Matrix<double, 8, 8> Q_;
    Eigen::Matrix4d R_;
    Eigen::Matrix<double, 8, 8> F_;
    Eigen::Matrix<double, 4, 8> H_;
};