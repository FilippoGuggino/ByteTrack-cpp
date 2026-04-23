#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score, bool is_blob, int label) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    rect_measured_(rect),
    has_fresh_measurement_(false),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0),
    age_(0),
    shadow_tracking_age_(0),
    blob_hits_(0),
    is_blob_track_(is_blob),
    last_ts_ns_(0),
    class_id_(label),
    consecutive_yolo_hits_(0),
    yolo_ever_matched_(false)
{
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::Rect<float>& byte_track::STrack::getMeasuredRect() const
{
    return rect_measured_;
}

bool byte_track::STrack::hasFreshMeasurement() const
{
    return has_fresh_measurement_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}

const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

size_t byte_track::STrack::getAge() const
{
    return age_;
}

size_t byte_track::STrack::getShadowTrackingAge() const
{
    return shadow_tracking_age_;
}

size_t byte_track::STrack::getBlobHits() const
{
    return blob_hits_;
}

bool byte_track::STrack::isBlobTrack() const
{
    return is_blob_track_;
}

int64_t byte_track::STrack::getLastTimestampNs() const
{
    return last_ts_ns_;
}

int byte_track::STrack::getClassId() const
{
    return class_id_;
}

size_t byte_track::STrack::getConsecutiveYoloHits() const
{
    return consecutive_yolo_hits_;
}

bool byte_track::STrack::getYoloEverMatched() const
{
    return yolo_ever_matched_;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id,
                                  int64_t ts_ns, bool is_blob)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());
    updateRect();
    rect_measured_ = rect_;
    has_fresh_measurement_ = true;

    state_ = STrackState::New;
    is_activated_ = false;
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
    age_ = 1;
    shadow_tracking_age_ = 0;
    blob_hits_ = is_blob ? 1 : 0;
    is_blob_track_ = is_blob;
    last_ts_ns_ = ts_ns;
    consecutive_yolo_hits_ = 0;
    // yolo_ever_matched_ and class_id_ are set from the constructor; do not reset here.
}

void byte_track::STrack::reActivate(const STrack& new_track, const size_t& frame_id,
                                    int64_t ts_ns, const int& new_track_id,
                                    size_t blob_to_yolo_transition_hits)
{
    rect_measured_ = new_track.getRect();
    has_fresh_measurement_ = true;
    if (is_blob_track_ && !new_track.isBlobTrack() && consecutive_yolo_hits_ == 0)
    {
        // First YOLO match on a blob-seeded track: Kalman scale (ar, h) is stuck at blob
        // dimensions and converges very slowly. Re-initiate from the YOLO measurement,
        // preserving position velocity so prediction remains useful.
        const float vx = mean_[4];
        const float vy = mean_[5];
        kalman_filter_.initiate(mean_, covariance_, new_track.getRect().getXyah());
        mean_[4] = vx;
        mean_[5] = vy;
    }
    else
    {
        kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
    }
    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
    shadow_tracking_age_ = 0;
    if (!new_track.isBlobTrack())
    {
        consecutive_yolo_hits_++;
        yolo_ever_matched_ = true;
        class_id_ = new_track.getClassId();
        // Only clear is_blob_track_ after enough consecutive YOLO hits so that the Kalman
        // filter has time to converge on YOLO box dimensions before switching to IoU matching.
        if (consecutive_yolo_hits_ >= blob_to_yolo_transition_hits)
        {
            is_blob_track_ = false;
        }
    }
    else
    {
        consecutive_yolo_hits_ = 0;
        // A blob matched a YOLO-originated track. The blob position is too noisy to trust
        // for velocity estimation; carrying stale velocity forward causes the Kalman to predict
        // far from where the target actually reappears. Reset velocity so the next prediction
        // stays at the current filtered position rather than overshooting.
        if (!is_blob_track_)
        {
            mean_[4] = 0.0f;
            mean_[5] = 0.0f;
        }
    }
    blob_hits_++;
    last_ts_ns_ = ts_ns;
}

void byte_track::STrack::predict(int64_t current_ts_ns)
{
    age_++;

    float dt = 1.0f;
    if (last_ts_ns_ > 0 && current_ts_ns > 0)
    {
        dt = static_cast<float>(current_ts_ns - last_ts_ns_) / 1e9f;
        dt = std::clamp(dt, 0.001f, 10.0f);
    }

    has_fresh_measurement_ = false;
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    kalman_filter_.predict(mean_, covariance_, dt);
}

void byte_track::STrack::update(const STrack& new_track, const size_t& frame_id, int64_t ts_ns,
                                size_t blob_to_yolo_transition_hits)
{
    rect_measured_ = new_track.getRect();
    has_fresh_measurement_ = true;
    if (is_blob_track_ && !new_track.isBlobTrack() && consecutive_yolo_hits_ == 0)
    {
        // First YOLO match on a blob-seeded track: Kalman scale (ar, h) is stuck at blob
        // dimensions and converges very slowly. Re-initiate from the YOLO measurement,
        // preserving position velocity so prediction remains useful.
        const float vx = mean_[4];
        const float vy = mean_[5];
        kalman_filter_.initiate(mean_, covariance_, new_track.getRect().getXyah());
        mean_[4] = vx;
        mean_[5] = vy;
    }
    else
    {
        kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
    }
    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
    shadow_tracking_age_ = 0;
    if (!new_track.isBlobTrack())
    {
        consecutive_yolo_hits_++;
        yolo_ever_matched_ = true;
        class_id_ = new_track.getClassId();
        // Only clear is_blob_track_ after enough consecutive YOLO hits so that the Kalman
        // filter has time to converge on YOLO box dimensions before switching to IoU matching.
        if (consecutive_yolo_hits_ >= blob_to_yolo_transition_hits)
        {
            is_blob_track_ = false;
        }
    }
    else
    {
        consecutive_yolo_hits_ = 0;
        // Same rationale as reActivate: reset velocity when a blob matches a YOLO track
        // to prevent stale blob-derived velocity from overshooting the next prediction.
        if (!is_blob_track_)
        {
            mean_[4] = 0.0f;
            mean_[5] = 0.0f;
        }
    }
    blob_hits_++;
    last_ts_ns_ = ts_ns;
}

void byte_track::STrack::promote()
{
    state_ = STrackState::Tracked;
    is_activated_ = true;
}

void byte_track::STrack::markAsShadow()
{
    state_ = STrackState::Shadow;
    shadow_tracking_age_++;
    consecutive_yolo_hits_ = 0;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}

float byte_track::STrack::getKalmanCx() const
{
    return mean_[0];
}

float byte_track::STrack::getKalmanCy() const
{
    return mean_[1];
}

float byte_track::STrack::getKalmanSigmaRadius() const
{
    return std::sqrt(covariance_(0, 0) + covariance_(1, 1));
}

void byte_track::STrack::applyEgoMotionCorrection(
    const Eigen::Matrix3f& R_delta, float fx, float fy, float cx, float cy)
{
    // Use the Kalman predicted center (mean_[0], mean_[1]) directly — rect_ may be
    // stale if predict() was called without a subsequent updateRect().
    const float u = mean_[0];
    const float v = mean_[1];

    // Unproject pixel center to a unit-depth bearing ray in the previous camera frame.
    const Eigen::Vector3f ray((u - cx) / fx, (v - cy) / fy, 1.f);

    // Rotate into the current camera frame (depth-independent for pure rotation).
    const Eigen::Vector3f ray_new = R_delta * ray;

    // Reproject to pixel coordinates and update the Kalman state.
    mean_[0] = fx * ray_new.x() / ray_new.z() + cx;
    mean_[1] = fy * ray_new.y() / ray_new.z() + cy;
    updateRect();
}
