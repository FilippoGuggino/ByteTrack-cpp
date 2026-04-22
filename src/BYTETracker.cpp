#include "ByteTrack/BYTETracker.h"

#include "Eigen/Dense"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

byte_track::BYTETracker::BYTETracker(const BYTETrackerConfig& config) :
    config_(config),
    frame_id_(0),
    track_id_count_(0),
    last_timestamp_ns_(0)
{
}

byte_track::BYTETracker::BYTETracker(const int& frame_rate,
                                     const int& track_buffer,
                                     const float& track_thresh,
                                     const float& high_thresh,
                                     const float& match_thresh) :
    BYTETracker(BYTETrackerConfig{
        .frame_rate = frame_rate,
        .track_buffer = track_buffer,
        .track_thresh = track_thresh,
        .high_thresh = high_thresh,
        .match_thresh = match_thresh
    })
{
}

byte_track::BYTETracker::~BYTETracker()
{
}

void byte_track::BYTETracker::setCameraParams(const CameraParams& params)
{
    camera_params_ = params;
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::update(const std::vector<Object>& objects, int64_t timestamp_ns)
{
    return update(objects, {}, EgoMotionData{}, timestamp_ns);
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::update(const std::vector<Object>& objects,
                                const std::vector<BlobObject>& blob_objects,
                                int64_t timestamp_ns)
{
    return update(objects, blob_objects, EgoMotionData{}, timestamp_ns);
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::update(const std::vector<Object>& objects,
                                const std::vector<BlobObject>& blob_objects,
                                const EgoMotionData& ego_motion,
                                int64_t timestamp_ns)
{
    ////////////////// Step 1: Setup frame & detections //////////////////
    frame_id_++;

    // Resolve timestamp: synthesize from frame count if none provided
    if (timestamp_ns <= 0)
    {
        timestamp_ns = static_cast<int64_t>(frame_id_) *
                       static_cast<int64_t>(1e9 / config_.frame_rate);
    }

    // Create STracks from model detections
    std::vector<STrackPtr> det_stracks;
    std::vector<STrackPtr> det_low_stracks;
    for (const auto& object : objects)
    {
        const auto strack = std::make_shared<STrack>(object.rect, object.prob, /*is_blob=*/false);
        if (object.prob >= config_.track_thresh)
        {
            det_stracks.push_back(strack);
        }
        else
        {
            det_low_stracks.push_back(strack);
        }
    }

    // Create STracks from blob detections (always treated as high-conf for init purposes)
    for (const auto& blob : blob_objects)
    {
        const float r = blob.radius;
        const Rect<float> rect(blob.x - r, blob.y - r, 2.0f * r, 2.0f * r);
        // Clamp response to [0,1] to use as confidence
        const float conf = std::min(1.0f, std::max(0.0f, blob.response));
        const auto strack = std::make_shared<STrack>(rect, conf, /*is_blob=*/true);
        // Blob detections always go into the high-conf pool if they pass high_thresh,
        // otherwise treat like low-conf model detections
        if (conf >= config_.track_thresh)
        {
            det_stracks.push_back(strack);
        }
        else
        {
            det_low_stracks.push_back(strack);
        }
    }

    // Partition existing tracked stracks by activation state
    std::vector<STrackPtr> active_stracks;    // confirmed (Tracked + New already confirmed)
    std::vector<STrackPtr> non_active_stracks; // New / probationary

    for (const auto& tracked_strack : tracked_stracks_)
    {
        if (!tracked_strack->isActivated())
        {
            non_active_stracks.push_back(tracked_strack);
        }
        else
        {
            active_stracks.push_back(tracked_strack);
        }
    }

    // Pool of tracks to predict: confirmed + shadow
    std::vector<STrackPtr> strack_pool = jointStracks(active_stracks, shadow_stracks_);

    // Predict all tracks with Kalman filter
    for (auto& strack : strack_pool)
    {
        strack->predict(timestamp_ns);
    }

    // Compensate camera rotation ego-motion: shift every predicted track position so that
    // static scene points stay put and only true object motion remains.
    applyEgoMotionCorrection(strack_pool, ego_motion);
    applyEgoMotionCorrection(non_active_stracks, ego_motion);

    ////////////////// Step 2: First association, with IoU (high-conf dets) //////////////////
    std::vector<STrackPtr> current_tracked_stracks;
    std::vector<STrackPtr> remain_tracked_stracks;
    std::vector<STrackPtr> remain_det_stracks;
    std::vector<STrackPtr> refind_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_detection_idx, unmatch_track_idx;

        const auto dists = calcIouDistance(strack_pool, det_stracks);
        linearAssignment(dists, strack_pool.size(), det_stracks.size(), config_.match_thresh,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            const auto track = strack_pool[match_idx[0]];
            const auto det = det_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_, timestamp_ns);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                // Shadow track re-matched
                track->reActivate(*det, frame_id_, timestamp_ns);
                refind_stracks.push_back(track);
            }
        }

        for (const auto& unmatch_idx : unmatch_detection_idx)
        {
            remain_det_stracks.push_back(det_stracks[unmatch_idx]);
        }

        for (const auto& unmatch_idx : unmatch_track_idx)
        {
            if (strack_pool[unmatch_idx]->getSTrackState() == STrackState::Tracked)
            {
                remain_tracked_stracks.push_back(strack_pool[unmatch_idx]);
            }
        }
    }

    ////////////////// Step 3: Second association, using low-conf dets //////////////////
    std::vector<STrackPtr> current_shadow_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_track_idx, unmatch_detection_idx;

        const auto dists = calcIouDistance(remain_tracked_stracks, det_low_stracks);
        linearAssignment(dists, remain_tracked_stracks.size(), det_low_stracks.size(), 0.5,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            const auto track = remain_tracked_stracks[match_idx[0]];
            const auto det = det_low_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_, timestamp_ns);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_, timestamp_ns);
                refind_stracks.push_back(track);
            }
        }

        for (const auto& unmatch_track : unmatch_track_idx)
        {
            const auto track = remain_tracked_stracks[unmatch_track];
            if (track->getSTrackState() != STrackState::Shadow)
            {
                track->markAsShadow();
                current_shadow_stracks.push_back(track);
            }
        }
    }

    ////////////////// Step 4: Init new stracks & manage probationary tracks //////////////////
    std::vector<STrackPtr> current_removed_stracks;

    {
        std::vector<int> unmatch_detection_idx;
        std::vector<int> unmatch_unconfirmed_idx;
        std::vector<std::vector<int>> matches_idx;

        // Associate probationary (non-activated) tracks with remaining detections
        const auto dists = calcIouDistance(non_active_stracks, remain_det_stracks);
        linearAssignment(dists, non_active_stracks.size(), remain_det_stracks.size(), 0.7,
                         matches_idx, unmatch_unconfirmed_idx, unmatch_detection_idx);

        for (const auto& match_idx : matches_idx)
        {
            const auto track = non_active_stracks[match_idx[0]];
            const auto det = remain_det_stracks[match_idx[1]];
            track->update(*det, frame_id_, timestamp_ns);
            current_tracked_stracks.push_back(track);
        }

        for (const auto& unmatch_idx : unmatch_unconfirmed_idx)
        {
            // Probationary track didn't match; enter shadow and check early termination
            const auto track = non_active_stracks[unmatch_idx];
            track->markAsShadow();
            if (track->getShadowTrackingAge() >=
                static_cast<size_t>(config_.early_termination_age))
            {
                track->markAsRemoved();
                current_removed_stracks.push_back(track);
            }
            else
            {
                current_shadow_stracks.push_back(track);
            }
        }

        // Seed new tracks from unmatched high-conf detections
        for (const auto& unmatch_idx : unmatch_detection_idx)
        {
            const auto track = remain_det_stracks[unmatch_idx];
            if (track->getScore() < config_.high_thresh)
            {
                continue;
            }
            track_id_count_++;
            track->activate(frame_id_, track_id_count_, timestamp_ns, track->isBlobTrack());
            current_tracked_stracks.push_back(track);
        }
    }

    ////////////////// Step 5: Probation check & shadow expiry //////////////////

    // Check probation for all newly-updated tracks (promote New → Tracked)
    for (const auto& track : current_tracked_stracks)
    {
        if (track->getSTrackState() == STrackState::New)
        {
            bool confirm = false;
            if (track->isBlobTrack())
            {
                confirm = track->getBlobHits() >=
                          static_cast<size_t>(config_.blob_probation_age);
            }
            else
            {
                confirm = track->getAge() >=
                          static_cast<size_t>(config_.probation_age);
            }
            if (confirm)
            {
                // Transition to Tracked / confirmed
                // is_activated_ was set in update(); state_ stays Tracked from update()
            }
        }
    }

    // Handle shadow track expiry
    for (const auto& shadow_strack : shadow_stracks_)
    {
        if (shadow_strack->getSTrackState() == STrackState::Shadow &&
            shadow_strack->getShadowTrackingAge() >
                static_cast<size_t>(config_.max_shadow_tracking_age))
        {
            shadow_strack->markAsRemoved();
            current_removed_stracks.push_back(shadow_strack);
        }
    }

    ////////////////// Step 6: Maintain track lists //////////////////
    tracked_stracks_ = jointStracks(current_tracked_stracks, refind_stracks);

    // Shadow pool: previous shadow tracks (minus any re-found or removed) + new shadow
    shadow_stracks_ = subStracks(
        jointStracks(subStracks(shadow_stracks_, tracked_stracks_), current_shadow_stracks),
        removed_stracks_);

    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    // Remove duplicates between tracked and shadow
    std::vector<STrackPtr> tracked_stracks_out, shadow_stracks_out;
    removeDuplicateStracks(tracked_stracks_, shadow_stracks_,
                           tracked_stracks_out, shadow_stracks_out);
    tracked_stracks_ = tracked_stracks_out;
    shadow_stracks_ = shadow_stracks_out;

    last_timestamp_ns_ = timestamp_ns;

    // Store drone orientation for the next frame's ego-motion computation.
    if (ego_motion.valid)
    {
        std::copy(std::begin(ego_motion.q_wb), std::end(ego_motion.q_wb), q_wb_prev_);
        has_prev_orientation_ = true;
    }

    ////////////////// Step 7: Collect output (confirmed Tracked only) //////////////////
    std::vector<STrackPtr> output_stracks;
    for (const auto& track : tracked_stracks_)
    {
        if (track->isActivated() && track->getSTrackState() == STrackState::Tracked)
        {
            // Check if probation is satisfied
            bool passes_probation = false;
            if (track->isBlobTrack())
            {
                passes_probation = track->getBlobHits() >=
                                   static_cast<size_t>(config_.blob_probation_age);
            }
            else
            {
                passes_probation = track->getAge() >=
                                   static_cast<size_t>(config_.probation_age);
            }
            if (passes_probation)
            {
                output_stracks.push_back(track);
            }
        }
    }

    return output_stracks;
}

void byte_track::BYTETracker::applyEgoMotionCorrection(
    std::vector<STrackPtr>& tracks, const EgoMotionData& ego_motion)
{
    if (!camera_params_.has_value() || !has_prev_orientation_ || !ego_motion.valid
        || tracks.empty())
    {
        return;
    }

    const auto& cam = *camera_params_;

    // q_wb: body-to-world quaternion (Eigen stores as w, x, y, z in constructor)
    // q_bc: camera-to-body quaternion
    // q_wc = q_wb * q_bc → rotates camera-frame vectors into world frame
    const Eigen::Quaternionf q_wb_prev(q_wb_prev_[0], q_wb_prev_[1],
                                       q_wb_prev_[2], q_wb_prev_[3]);
    const Eigen::Quaternionf q_wb_curr(ego_motion.q_wb[0], ego_motion.q_wb[1],
                                       ego_motion.q_wb[2], ego_motion.q_wb[3]);
    const Eigen::Quaternionf q_bc(cam.q_bc[0], cam.q_bc[1], cam.q_bc[2], cam.q_bc[3]);

    // R_wc: rotation matrix that transforms camera-frame vectors into world frame.
    const Eigen::Matrix3f R_wc_prev = (q_wb_prev * q_bc).toRotationMatrix();
    const Eigen::Matrix3f R_wc_curr = (q_wb_curr * q_bc).toRotationMatrix();

    // R_delta maps a bearing ray from the previous camera frame into the current one.
    // For a static world point its image-plane position changes by exactly this rotation.
    const Eigen::Matrix3f R_delta = R_wc_curr.transpose() * R_wc_prev;

    for (auto& track : tracks)
    {
        track->applyEgoMotionCorrection(R_delta, cam.fx, cam.fy, cam.cx, cam.cy);
    }
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::getAllTracks() const
{
    std::vector<STrackPtr> all;
    for (const auto& t : tracked_stracks_) all.push_back(t);
    for (const auto& t : shadow_stracks_) all.push_back(t);
    return all;
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::getTentativeTracks() const
{
    std::vector<STrackPtr> result;
    for (const auto& t : tracked_stracks_)
    {
        if (t->getSTrackState() == STrackState::New)
        {
            result.push_back(t);
        }
    }
    return result;
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::jointStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, int> exists;
    std::vector<STrackPtr> res;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        exists.emplace(a_tlist[i]->getTrackId(), 1);
        res.push_back(a_tlist[i]);
    }
    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(b_tlist[i]);
        }
    }
    return res;
}

std::vector<byte_track::BYTETracker::STrackPtr>
byte_track::BYTETracker::subStracks(const std::vector<STrackPtr> &a_tlist,
                                    const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, STrackPtr> stracks;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        stracks.emplace(a_tlist[i]->getTrackId(), a_tlist[i]);
    }

    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    std::vector<STrackPtr> res;
    std::map<int, STrackPtr>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void byte_track::BYTETracker::removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                                     const std::vector<STrackPtr> &b_stracks,
                                                     std::vector<STrackPtr> &a_res,
                                                     std::vector<STrackPtr> &b_res) const
{
    const auto ious = calcIouDistance(a_stracks, b_stracks);

    std::vector<std::pair<size_t, size_t>> overlapping_combinations;
    for (size_t i = 0; i < ious.size(); i++)
    {
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            if (ious[i][j] < 0.15)
            {
                overlapping_combinations.emplace_back(i, j);
            }
        }
    }

    std::vector<bool> a_overlapping(a_stracks.size(), false),
                      b_overlapping(b_stracks.size(), false);
    for (const auto &[a_idx, b_idx] : overlapping_combinations)
    {
        const int timep = a_stracks[a_idx]->getFrameId() - a_stracks[a_idx]->getStartFrameId();
        const int timeq = b_stracks[b_idx]->getFrameId() - b_stracks[b_idx]->getStartFrameId();
        if (timep > timeq)
        {
            b_overlapping[b_idx] = true;
        }
        else
        {
            a_overlapping[a_idx] = true;
        }
    }

    for (size_t ai = 0; ai < a_stracks.size(); ai++)
    {
        if (!a_overlapping[ai])
        {
            a_res.push_back(a_stracks[ai]);
        }
    }

    for (size_t bi = 0; bi < b_stracks.size(); bi++)
    {
        if (!b_overlapping[bi])
        {
            b_res.push_back(b_stracks[bi]);
        }
    }
}

void byte_track::BYTETracker::linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                                               const int &cost_matrix_size,
                                               const int &cost_matrix_size_size,
                                               const float &thresh,
                                               std::vector<std::vector<int>> &matches,
                                               std::vector<int> &a_unmatched,
                                               std::vector<int> &b_unmatched) const
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            a_unmatched.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            b_unmatched.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol; std::vector<int> colsol;
    execLapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (size_t i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            a_unmatched.push_back(i);
        }
    }

    for (size_t i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            b_unmatched.push_back(i);
        }
    }
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcIous(
    const std::vector<Rect<float>> &a_rect,
    const std::vector<Rect<float>> &b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.size() * b_rect.size() == 0)
    {
        return ious;
    }

    ious.resize(a_rect.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        ious[i].resize(b_rect.size());
    }

    for (size_t bi = 0; bi < b_rect.size(); bi++)
    {
        for (size_t ai = 0; ai < a_rect.size(); ai++)
        {
            ious[ai][bi] = b_rect[bi].calcIoU(a_rect[ai]);
        }
    }
    return ious;
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcIouDistance(
    const std::vector<STrackPtr> &a_tracks,
    const std::vector<STrackPtr> &b_tracks) const
{
    std::vector<byte_track::Rect<float>> a_rects, b_rects;
    for (size_t i = 0; i < a_tracks.size(); i++)
    {
        a_rects.push_back(a_tracks[i]->getRect());
    }

    for (size_t i = 0; i < b_tracks.size(); i++)
    {
        b_rects.push_back(b_tracks[i]->getRect());
    }

    const auto ious = calcIous(a_rects, b_rects);

    std::vector<std::vector<float>> cost_matrix;
    for (size_t i = 0; i < ious.size(); i++)
    {
        std::vector<float> iou;
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            iou.push_back(1 - ious[i][j]);
        }
        cost_matrix.push_back(iou);
    }

    return cost_matrix;
}

double byte_track::BYTETracker::execLapjv(const std::vector<std::vector<float>> &cost,
                                          std::vector<int> &rowsol,
                                          std::vector<int> &colsol,
                                          bool extend_cost,
                                          float cost_limit,
                                          bool return_cost) const
{
    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            throw std::runtime_error("The `extend_cost` variable should set True");
        }
    }

    if (extend_cost || cost_limit < std::numeric_limits<float>::max())
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (size_t i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < std::numeric_limits<float>::max())
        {
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (size_t i = 0; i < cost_c.size(); i++)
            {
                for (size_t j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        throw std::runtime_error("The result of lapjv_internal() is invalid.");
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        if (return_cost)
        {
            for (size_t i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    }
    else if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}
