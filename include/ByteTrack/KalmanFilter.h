#pragma once

#include "Eigen/Dense"

#include "ByteTrack/CameraParams.h"
#include "ByteTrack/Rect.h"

namespace byte_track
{
// Extended Kalman Filter for 3D tracking in camera frame.
//
// State (7D):     [X_c, Y_c, Z_c, vX_c, vY_c, vZ_c, size]
// Measurement(3D): [cx_px, cy_px, h_px]  (pixel centre + apparent height)
//
// Observation function h(x) is the perspective projection:
//   h = [fx*X/Z + cx,  fy*Y/Z + cy,  fy*size/Z]
// which is non-linear in Z — this is what makes it an EKF.
class KalmanFilter
{
public:
    using StateMean = Eigen::Matrix<float, 1, 7, Eigen::RowMajor>;
    using StateCov  = Eigen::Matrix<float, 7, 7, Eigen::RowMajor>;

    using MeasVec      = Eigen::Matrix<float, 1, 3, Eigen::RowMajor>;
    using MeasCov      = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
    using MeasJacobian = Eigen::Matrix<float, 3, 7, Eigen::RowMajor>;

    // Noise spectral densities for Q and pixel std-devs for R.
    // q_pos/q_vel: lateral (X,Y) position/velocity noise [m²/s or m²/s³]
    // q_pos_z/q_vel_z: depth (Z) noise (higher — depth is harder to observe)
    // q_size: physical size random-walk noise
    // r_px: pixel noise for cx, cy measurements
    // r_h_px: pixel noise for apparent-height measurement
    KalmanFilter(float q_pos   = 0.5f,
                 float q_vel   = 2.0f,
                 float q_pos_z = 2.0f,
                 float q_vel_z = 5.0f,
                 float q_size  = 0.01f,
                 float r_px    = 1.5f,
                 float r_h_px  = 3.0f);

    // Initialise state and covariance from a bounding-box detection and known physical size.
    void initiate(StateMean& mean, StateCov& covariance,
                  const CameraParams& cam,
                  const Xyah<float>& xyah,
                  float expected_size_m);

    // Predict state forward by dt seconds (constant-velocity model).
    void predict(StateMean& mean, StateCov& covariance, float dt);

    // EKF update: linearise h(x) at current mean and assimilate detection.
    void update(StateMean& mean, StateCov& covariance,
                const CameraParams& cam,
                const Xyah<float>& xyah);

private:
    float q_pos_, q_vel_, q_pos_z_, q_vel_z_, q_size_;
    float r_px_, r_h_px_;

    Eigen::Matrix<float, 7, 7, Eigen::RowMajor> buildF(float dt) const;
    Eigen::Matrix<float, 7, 7, Eigen::RowMajor> buildQ(float dt) const;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> buildR() const;

    // Evaluate the non-linear observation function h(x).
    MeasVec measurementFunction(const StateMean& mean, const CameraParams& cam) const;

    // Compute the 3×7 Jacobian H = ∂h/∂x at the current state estimate.
    MeasJacobian measurementJacobian(const StateMean& mean, const CameraParams& cam) const;
};
}
