#include "ByteTrack/KalmanFilter.h"

#include <cmath>

byte_track::KalmanFilter::KalmanFilter(float q_pos, float q_vel,
                                       float q_pos_z, float q_vel_z,
                                       float q_size,
                                       float r_px, float r_h_px) :
    q_pos_(q_pos),
    q_vel_(q_vel),
    q_pos_z_(q_pos_z),
    q_vel_z_(q_vel_z),
    q_size_(q_size),
    r_px_(r_px),
    r_h_px_(r_h_px)
{
}

Eigen::Matrix<float, 7, 7, Eigen::RowMajor>
byte_track::KalmanFilter::buildF(float dt) const
{
    Eigen::Matrix<float, 7, 7, Eigen::RowMajor> F =
        Eigen::Matrix<float, 7, 7, Eigen::RowMajor>::Identity();
    // position += velocity * dt  (top-right 3×3 block)
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    return F;
}

Eigen::Matrix<float, 7, 7, Eigen::RowMajor>
byte_track::KalmanFilter::buildQ(float dt) const
{
    Eigen::Matrix<float, 7, 7, Eigen::RowMajor> Q =
        Eigen::Matrix<float, 7, 7, Eigen::RowMajor>::Zero();
    Q(0, 0) = q_pos_  * dt;
    Q(1, 1) = q_pos_  * dt;
    Q(2, 2) = q_pos_z_ * dt;
    Q(3, 3) = q_vel_  * dt;
    Q(4, 4) = q_vel_  * dt;
    Q(5, 5) = q_vel_z_ * dt;
    Q(6, 6) = q_size_ * dt;
    return Q;
}

Eigen::Matrix<float, 3, 3, Eigen::RowMajor>
byte_track::KalmanFilter::buildR() const
{
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Zero();
    R(0, 0) = r_px_   * r_px_;
    R(1, 1) = r_px_   * r_px_;
    R(2, 2) = r_h_px_ * r_h_px_;
    return R;
}

byte_track::KalmanFilter::MeasVec
byte_track::KalmanFilter::measurementFunction(const StateMean& mean,
                                              const CameraParams& cam) const
{
    const float X = mean(0), Y = mean(1), Z = mean(2), s = mean(6);
    MeasVec h;
    h(0) = cam.fx * X / Z + cam.cx;
    h(1) = cam.fy * Y / Z + cam.cy;
    h(2) = cam.fy * s / Z;
    return h;
}

byte_track::KalmanFilter::MeasJacobian
byte_track::KalmanFilter::measurementJacobian(const StateMean& mean,
                                              const CameraParams& cam) const
{
    const float X = mean(0), Y = mean(1), Z = mean(2), s = mean(6);
    const float Z2 = Z * Z;

    MeasJacobian H = MeasJacobian::Zero();
    // ∂(fx·X/Z + cx)/∂x
    H(0, 0) =  cam.fx / Z;
    H(0, 2) = -cam.fx * X / Z2;
    // ∂(fy·Y/Z + cy)/∂x
    H(1, 1) =  cam.fy / Z;
    H(1, 2) = -cam.fy * Y / Z2;
    // ∂(fy·s/Z)/∂x
    H(2, 2) = -cam.fy * s / Z2;
    H(2, 6) =  cam.fy / Z;
    return H;
}

void byte_track::KalmanFilter::initiate(StateMean& mean, StateCov& covariance,
                                        const CameraParams& cam,
                                        const Xyah<float>& xyah,
                                        float expected_size_m)
{
    const float cx_px = xyah(0);
    const float cy_px = xyah(1);
    const float h_px  = xyah(3);

    // Estimate depth from apparent height and known physical size.
    const float Z0   = cam.fy * expected_size_m / h_px;
    const float X0   = (cx_px - cam.cx) / cam.fx * Z0;
    const float Y0   = (cy_px - cam.cy) / cam.fy * Z0;

    mean = StateMean::Zero();
    mean(0) = X0;
    mean(1) = Y0;
    mean(2) = Z0;
    mean(6) = expected_size_m;

    // Initial covariance: lateral position well-known from pixels,
    // depth more uncertain, velocity completely unknown.
    const float sig_xy   = 0.10f * Z0;
    const float sig_z    = 0.30f * Z0;
    const float sig_vel  = 2.0f;
    const float sig_size = 0.10f * expected_size_m;

    StateMean std_vec;
    std_vec(0) = sig_xy;
    std_vec(1) = sig_xy;
    std_vec(2) = sig_z;
    std_vec(3) = sig_vel;
    std_vec(4) = sig_vel;
    std_vec(5) = sig_vel;
    std_vec(6) = sig_size;

    covariance = std_vec.array().square().matrix().asDiagonal();
}

void byte_track::KalmanFilter::predict(StateMean& mean, StateCov& covariance, float dt)
{
    const auto F = buildF(dt);
    const auto Q = buildQ(dt);
    mean       = (F * mean.transpose()).transpose();
    covariance = F * covariance * F.transpose() + Q;
}

void byte_track::KalmanFilter::update(StateMean& mean, StateCov& covariance,
                                      const CameraParams& cam,
                                      const Xyah<float>& xyah)
{
    // Build measurement vector [cx_px, cy_px, h_px].
    MeasVec z;
    z(0) = xyah(0);
    z(1) = xyah(1);
    z(2) = xyah(3);

    const MeasVec      h = measurementFunction(mean, cam);
    const MeasJacobian H = measurementJacobian(mean, cam);
    const MeasCov      R = buildR();

    // Innovation covariance S = H·P·Hᵀ + R
    const MeasCov S = H * covariance * H.transpose() + R;

    // Kalman gain K = P·Hᵀ·S⁻¹  →  solve Sᵀ·Kᵀ = H·Pᵀ
    const Eigen::Matrix<float, 7, 3> K =
        (S.llt().solve((covariance * H.transpose()).transpose())).transpose();

    // State and covariance update.
    const MeasVec innovation = z - h;
    mean       = (mean.transpose() + K * innovation.transpose()).transpose();
    covariance = covariance - K * S * K.transpose();
}
