#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score, bool is_blob, int label) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    cam_config_(std::nullopt),
    expected_size_m_(0.5f),
    stored_ar_(1.0f),
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

void byte_track::STrack::setEkfConfig(const CameraParams& cam, float expected_size_m)
{
    cam_config_      = cam;
    expected_size_m_ = expected_size_m;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id,
                                  int64_t ts_ns, bool is_blob)
{
    const auto xyah = rect_.getXyah();
    stored_ar_ = xyah(2);

    if (cam_config_.has_value())
    {
        kalman_filter_.initiate(mean_, covariance_, *cam_config_, xyah, expected_size_m_);
    }
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

    const auto xyah = new_track.getRect().getXyah();
    stored_ar_ = xyah(2);

    if (cam_config_.has_value())
    {
        if (is_blob_track_ && !new_track.isBlobTrack() && consecutive_yolo_hits_ == 0)
        {
            // First YOLO match on a blob-seeded track: re-initiate from YOLO measurement,
            // preserving 3D velocity so prediction remains useful.
            const float vx = mean_(3);
            const float vy = mean_(4);
            const float vz = mean_(5);
            kalman_filter_.initiate(mean_, covariance_, *cam_config_, xyah, expected_size_m_);
            mean_(3) = vx;
            mean_(4) = vy;
            mean_(5) = vz;
        }
        else
        {
            kalman_filter_.update(mean_, covariance_, *cam_config_, xyah);
        }
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
        // A blob matched a YOLO-originated track. Reset velocity so the next prediction
        // stays at the current filtered position rather than overshooting.
        if (!is_blob_track_)
        {
            mean_(3) = 0.0f;
            mean_(4) = 0.0f;
            mean_(5) = 0.0f;
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
    kalman_filter_.predict(mean_, covariance_, dt);
}

void byte_track::STrack::update(const STrack& new_track, const size_t& frame_id, int64_t ts_ns,
                                size_t blob_to_yolo_transition_hits)
{
    rect_measured_ = new_track.getRect();
    has_fresh_measurement_ = true;

    const auto xyah = new_track.getRect().getXyah();
    stored_ar_ = xyah(2);

    if (cam_config_.has_value())
    {
        if (is_blob_track_ && !new_track.isBlobTrack() && consecutive_yolo_hits_ == 0)
        {
            // First YOLO match on a blob-seeded track: re-initiate from YOLO measurement,
            // preserving 3D velocity so prediction remains useful.
            const float vx = mean_(3);
            const float vy = mean_(4);
            const float vz = mean_(5);
            kalman_filter_.initiate(mean_, covariance_, *cam_config_, xyah, expected_size_m_);
            mean_(3) = vx;
            mean_(4) = vy;
            mean_(5) = vz;
        }
        else
        {
            kalman_filter_.update(mean_, covariance_, *cam_config_, xyah);
        }
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
        // Same rationale as reActivate: reset velocity when a blob matches a YOLO track.
        if (!is_blob_track_)
        {
            mean_(3) = 0.0f;
            mean_(4) = 0.0f;
            mean_(5) = 0.0f;
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
    if (!cam_config_.has_value())
    {
        return;
    }

    const auto& cam = *cam_config_;
    const float X   = mean_(0);
    const float Y   = mean_(1);
    const float Z   = mean_(2);
    const float s   = mean_(6);

    if (Z <= 0.0f) { return; }

    const float cx_px = cam.fx * X / Z + cam.cx;
    const float cy_px = cam.fy * Y / Z + cam.cy;
    const float h_px  = cam.fy * s / Z;
    const float w_px  = stored_ar_ * h_px;

    rect_.x()      = cx_px - w_px * 0.5f;
    rect_.y()      = cy_px - h_px * 0.5f;
    rect_.width()  = w_px;
    rect_.height() = h_px;
}

void byte_track::STrack::applyEgoMotionCorrection(
    const Eigen::Matrix3f& R_delta, float /*fx*/, float /*fy*/, float /*cx*/, float /*cy*/)
{
    if (!cam_config_.has_value()) { return; }

    // Rotate position and velocity directly in 3D camera frame.
    const Eigen::Vector3f pos(mean_(0), mean_(1), mean_(2));
    const Eigen::Vector3f vel(mean_(3), mean_(4), mean_(5));

    const Eigen::Vector3f pos_new = R_delta * pos;
    const Eigen::Vector3f vel_new = R_delta * vel;

    mean_(0) = pos_new.x();
    mean_(1) = pos_new.y();
    mean_(2) = pos_new.z();
    mean_(3) = vel_new.x();
    mean_(4) = vel_new.y();
    mean_(5) = vel_new.z();

    updateRect();
}
