#pragma once

#include "ByteTrack/Rect.h"
#include "ByteTrack/KalmanFilter.h"
#include "Eigen/Dense"

#include <cstddef>
#include <cstdint>

namespace byte_track
{
enum class STrackState {
    New = 0,       // Newly created; in probation period, not yet output
    Tracked = 1,   // Confirmed and actively matched; included in output
    Shadow = 2,    // Confirmed but unmatched; grace period before removal
    Removed = 3,   // Expired; no longer tracked
};

class STrack
{
public:
    STrack(const Rect<float>& rect, const float& score, bool is_blob = false, int label = -1);
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    // New getters
    size_t getAge() const;
    size_t getShadowTrackingAge() const;
    size_t getBlobHits() const;
    bool isBlobTrack() const;
    int64_t getLastTimestampNs() const;
    int getClassId() const;
    size_t getConsecutiveYoloHits() const;
    bool getYoloEverMatched() const;

    void activate(const size_t& frame_id, const size_t& track_id,
                  int64_t ts_ns, bool is_blob = false);
    void reActivate(const STrack& new_track, const size_t& frame_id,
                    int64_t ts_ns, const int& new_track_id = -1);

    // Predict state forward using Kalman filter; dt derived from ts_ns vs last_ts_ns_
    void predict(int64_t current_ts_ns);

    void update(const STrack& new_track, const size_t& frame_id, int64_t ts_ns);

    void promote();  // Transition New → Tracked once probation is satisfied
    void markAsShadow();
    void markAsRemoved();

    // Apply camera rotation ego-motion correction to the Kalman state position.
    // R_delta rotates a bearing ray from the previous camera frame into the current one.
    // The corrected center is reprojected using the supplied intrinsics and rect_ is synced.
    void applyEgoMotionCorrection(const Eigen::Matrix3f& R_delta,
                                  float fx, float fy, float cx, float cy);

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    bool is_activated_;
    float score_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;

    // Probation / shadow tracking
    size_t age_;
    size_t shadow_tracking_age_;
    size_t blob_hits_;
    bool is_blob_track_;
    int64_t last_ts_ns_;  // ns timestamp of last successful match

    // Detection class and YOLO match tracking
    int class_id_;                 // -1 = unknown (blob-seeded)
    size_t consecutive_yolo_hits_; // consecutive frames matched by model; reset on miss or blob match
    bool yolo_ever_matched_;       // set on first model match, never cleared

    void updateRect();
};
}
