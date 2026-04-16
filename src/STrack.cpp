#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score, bool is_blob) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
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
    last_ts_ns_(0)
{
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
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

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id,
                                  int64_t ts_ns, bool is_blob)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());
    updateRect();

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
}

void byte_track::STrack::reActivate(const STrack& new_track, const size_t& frame_id,
                                    int64_t ts_ns, const int& new_track_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
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
        is_blob_track_ = false;
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

    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    kalman_filter_.predict(mean_, covariance_, dt);
}

void byte_track::STrack::update(const STrack& new_track, const size_t& frame_id, int64_t ts_ns)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());
    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
    shadow_tracking_age_ = 0;
    if (!new_track.isBlobTrack())
    {
        is_blob_track_ = false;
    }
    blob_hits_++;
    last_ts_ns_ = ts_ns;
}

void byte_track::STrack::markAsShadow()
{
    state_ = STrackState::Shadow;
    shadow_tracking_age_++;
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
