#pragma once

#include "ByteTrack/STrack.h"
#include "ByteTrack/lapjv.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/BlobObject.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace byte_track
{
// Intrinsic and extrinsic camera parameters needed for ego-motion compensation.
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

struct BYTETrackerConfig
{
    // Frame rate used to synthesize timestamps when none are provided (timestamp_ns == 0)
    int frame_rate = 30;
    // Number of frames to keep shadow tracks before removing them
    int track_buffer = 30;

    // Confidence threshold separating high-conf and low-conf detections
    float track_thresh = 0.5f;
    // Minimum confidence for a detection to seed a new track
    float high_thresh = 0.6f;
    // IoU threshold for the first-stage association
    float match_thresh = 0.8f;

    // Frames a model-seeded track must be matched before becoming Tracked (output)
    int probation_age = 1;
    // Matches a blob-seeded track must accumulate before becoming Tracked (output)
    int blob_probation_age = 5;
    // Consecutive frames without a match a Shadow track may persist before removal
    int max_shadow_tracking_age = 3;
    // New (probationary) tracks are removed after this many frames in shadow
    int early_termination_age = 1;
    // Max center-point distance (pixels) for matching blob tracks; beyond this, cost = 1.0
    float blob_match_max_dist_px = 100.0f;
};

class BYTETracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    explicit BYTETracker(const BYTETrackerConfig& config = BYTETrackerConfig());

    // Backward-compatible constructor (delegates to config-based one)
    BYTETracker(const int& frame_rate,
                const int& track_buffer,
                const float& track_thresh = 0.5f,
                const float& high_thresh = 0.6f,
                const float& match_thresh = 0.8f);

    ~BYTETracker();

    // Configure camera intrinsics and body-to-camera extrinsic for ego-motion compensation.
    // Must be called before passing EgoMotionData to update(); otherwise ego-motion is ignored.
    void setCameraParams(const CameraParams& params);

    // Primary update with ego-motion compensation: model + blob detections + drone orientation.
    std::vector<STrackPtr> update(const std::vector<Object>& objects,
                                  const std::vector<BlobObject>& blob_objects,
                                  const EgoMotionData& ego_motion,
                                  int64_t timestamp_ns = 0);

    // Backward-compatible: model detections + blob detections, no ego-motion.
    std::vector<STrackPtr> update(const std::vector<Object>& objects,
                                  const std::vector<BlobObject>& blob_objects,
                                  int64_t timestamp_ns = 0);

    // Backward-compatible: model detections only, no ego-motion.
    std::vector<STrackPtr> update(const std::vector<Object>& objects,
                                  int64_t timestamp_ns = 0);

    // For testing / debugging: returns all tracks regardless of state
    std::vector<STrackPtr> getAllTracks() const;

    // For testing / debugging: returns only New (probationary) tracks
    std::vector<STrackPtr> getTentativeTracks() const;

private:
    // Rotate each track's Kalman-predicted position to compensate for camera ego-motion.
    // Uses consecutive drone orientation quaternions to compute the inter-frame camera rotation
    // and applies the resulting pixel shift to every track in the list.
    void applyEgoMotionCorrection(std::vector<STrackPtr>& tracks,
                                  const EgoMotionData& ego_motion);

    std::optional<CameraParams> camera_params_;
    bool has_prev_orientation_ = false;
    float q_wb_prev_[4] = {1.f, 0.f, 0.f, 0.f};  // w, x, y, z

    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks) const;

    // Blob-aware distance: for any pair involving a blob track, uses Euclidean center-point
    // distance normalized by blob_match_max_dist_px; otherwise falls back to 1 - IoU.
    std::vector<std::vector<float>> calcMatchingDistance(const std::vector<STrackPtr> &a_tracks,
                                                         const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

private:
    BYTETrackerConfig config_;

    size_t frame_id_;
    size_t track_id_count_;
    int64_t last_timestamp_ns_;

    std::vector<STrackPtr> tracked_stracks_;   // New + Tracked
    std::vector<STrackPtr> shadow_stracks_;    // Shadow (grace period)
    std::vector<STrackPtr> removed_stracks_;   // Permanently removed
};
}
