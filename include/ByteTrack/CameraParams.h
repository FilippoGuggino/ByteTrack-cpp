#pragma once

namespace byte_track
{
// Intrinsic and extrinsic camera parameters used by the EKF and ego-motion compensation.
struct CameraParams {
    float fx = 0.f, fy = 0.f;  // Focal lengths in pixels
    float cx = 0.f, cy = 0.f;  // Principal point in pixels
    // Quaternion rotating camera-frame vectors into drone body frame (q_bc): w, x, y, z
    float q_bc[4] = {1.f, 0.f, 0.f, 0.f};
};

// Per-frame drone orientation used to compensate camera ego-motion in the tracker.
// Matches the PX4 VehicleOdometry quaternion convention (body-to-world, NED world frame).
struct EgoMotionData {
    // Body-to-world orientation quaternion (q_wb): w, x, y, z
    float q_wb[4] = {1.f, 0.f, 0.f, 0.f};
    // Set to true only when q_wb contains a valid measurement.
    bool valid = false;
};
}
