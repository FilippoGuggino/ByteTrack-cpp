#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/BlobObject.h"

#include "gtest/gtest.h"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
#include "boost/optional.hpp"

#include <cstddef>
#include <cstdint>

namespace
{
    constexpr double EPS = 1e-2;

    const std::string D_RESULTS_FILE = "detection_results.json";
    const std::string T_RESULTS_FILE = "tracking_results.json";

    // key: track_id, value: rect of tracking object
    using BYTETrackerOut = std::map<size_t, byte_track::Rect<float>>;

    template <typename T>
    T get_data(const boost::property_tree::ptree &pt, const std::string &key)
    {
        T ret;
        if (boost::optional<T> data = pt.get_optional<T>(key))
        {
            ret = data.get();
        }
        else
        {
            throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
        }
        return ret;
    }

    std::map<size_t, std::vector<byte_track::Object>> get_inputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, std::vector<byte_track::Object>> inputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto prob = get_data<float>(result, "prob");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(inputs_ref)::iterator itr = inputs_ref.find(frame_id);
            if (itr != inputs_ref.end())
            {
                itr->second.emplace_back(byte_track::Rect(x, y, width, height), 0, prob);
            }
            else
            {
                std::vector<byte_track::Object> v(1, {byte_track::Rect(x, y, width, height), 0, prob});
                inputs_ref.emplace_hint(inputs_ref.end(), frame_id, v);
            }
        }
        return inputs_ref;
    }

    std::map<size_t, BYTETrackerOut> get_outputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, BYTETrackerOut> outputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto track_id = get_data<int>(result, "track_id");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(outputs_ref)::iterator itr = outputs_ref.find(frame_id);
            if (itr != outputs_ref.end())
            {
                itr->second.emplace(track_id, byte_track::Rect<float>(x, y, width, height));
            }
            else
            {
                BYTETrackerOut v{
                    {track_id, byte_track::Rect<float>(x, y, width, height)},
                };
                outputs_ref.emplace_hint(outputs_ref.end(), frame_id, v);
            }
        }
        return outputs_ref;
    }
}

TEST(ByteTrack, BYTETracker)
{
    boost::property_tree::ptree pt_d_results;
    boost::property_tree::read_json(D_RESULTS_FILE, pt_d_results);

    boost::property_tree::ptree pt_t_results;
    boost::property_tree::read_json(T_RESULTS_FILE, pt_t_results);

    try
    {
        // Get infomation of reference data
        const auto detection_results_name = get_data<std::string>(pt_d_results, "name");
        const auto tracking_results_name = get_data<std::string>(pt_t_results, "name");
        const auto fps = get_data<int>(pt_d_results, "fps");
        const auto track_buffer = get_data<int>(pt_d_results, "track_buffer");

        if (detection_results_name != tracking_results_name)
        {
            throw std::runtime_error("The name of the tests are different: [detection_results_name: " + detection_results_name + 
                                     ", tracking_results_name: " + tracking_results_name + "]");
        }

        // Get input reference data from D_RESULTS_FILE
        const auto inputs_ref = get_inputs_ref(pt_d_results);

        // Get output reference data from T_RESULTS_FILE
        auto outputs_ref = get_outputs_ref(pt_t_results);

        // Test BYTETracker::update()
        byte_track::BYTETracker tracker(fps, track_buffer);
        for (const auto &[frame_id, objects] : inputs_ref)
        {
            const auto outputs = tracker.update(objects);

            // Verify between the reference data and the output of the BYTETracker impl
            EXPECT_EQ(outputs.size(), outputs_ref[frame_id].size());
            for (const auto &outputs_per_frame : outputs)
            {
                const auto &rect = outputs_per_frame->getRect();
                const auto &track_id = outputs_per_frame->getTrackId();
                const auto &ref = outputs_ref[frame_id][track_id];
                EXPECT_NEAR(ref.x(), rect.x(), EPS);
                EXPECT_NEAR(ref.y(), rect.y(), EPS);
                EXPECT_NEAR(ref.width(), rect.width(), EPS);
                EXPECT_NEAR(ref.height(), rect.height(), EPS);
            }
        }
    }
    catch (const std::exception &e)
    {
        FAIL() << e.what();
    }
}

// ---------------------------------------------------------------------------
// Helpers for synthetic tests
// ---------------------------------------------------------------------------
namespace
{
    // Build a simple Object (model detection) at given position
    byte_track::Object makeObject(float x, float y, float w, float h, float prob = 0.9f)
    {
        return byte_track::Object(byte_track::Rect<float>(x, y, w, h), 0, prob);
    }

    // Build a BlobObject at given center with given radius
    byte_track::BlobObject makeBlob(float cx, float cy, float radius, float response = 1.0f)
    {
        byte_track::BlobObject b;
        b.x = cx;
        b.y = cy;
        b.radius = radius;
        b.response = response;
        return b;
    }

    constexpr int64_t MS_NS = 1'000'000;   // 1 ms in nanoseconds
    constexpr int64_t SEC_NS = 1'000'000'000LL; // 1 s in nanoseconds
}

// ---------------------------------------------------------------------------
// Test 1: Variable frame rate — tracks survive irregular timestamp intervals
// ---------------------------------------------------------------------------
TEST(ByteTrack, VariableFrameRate)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.frame_rate = 30;
    cfg.probation_age = 1;
    cfg.max_shadow_tracking_age = 5;
    byte_track::BYTETracker tracker(cfg);

    const float x = 100.f, y = 100.f, w = 50.f, h = 80.f;

    // Frame 0 at t=0ms
    {
        const auto out = tracker.update({makeObject(x, y, w, h)}, {}, 0 * MS_NS);
        // After frame 0: probation_age=1, age=1 => should appear immediately
        EXPECT_EQ(out.size(), 1u);
    }

    // Frame 1 at t=100ms (normal interval)
    size_t track_id = 0;
    {
        const auto out = tracker.update({makeObject(x, y, w, h)}, {}, 100 * MS_NS);
        EXPECT_EQ(out.size(), 1u);
        if (!out.empty()) track_id = out[0]->getTrackId();
    }

    // Frame 2 at t=300ms (longer gap — 200ms)
    {
        const auto out = tracker.update({makeObject(x, y, w, h)}, {}, 300 * MS_NS);
        EXPECT_EQ(out.size(), 1u);
        if (!out.empty()) EXPECT_EQ(out[0]->getTrackId(), track_id);
    }

    // Frame 3 at t=500ms (another long gap)
    {
        const auto out = tracker.update({makeObject(x + 2, y + 2, w, h)}, {}, 500 * MS_NS);
        EXPECT_EQ(out.size(), 1u);
        if (!out.empty())
        {
            EXPECT_EQ(out[0]->getTrackId(), track_id);
            // Kalman state must not be NaN
            EXPECT_FALSE(std::isnan(out[0]->getRect().x()));
            EXPECT_FALSE(std::isnan(out[0]->getRect().y()));
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Shadow tracking — confirmed track persists through missed frames
//         then is removed after grace period expires
// ---------------------------------------------------------------------------
TEST(ByteTrack, ShadowTracking)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.probation_age = 1;
    cfg.max_shadow_tracking_age = 2;  // remove after 2 missed frames
    byte_track::BYTETracker tracker(cfg);

    const auto obj = makeObject(200.f, 200.f, 60.f, 90.f);

    // Frame 1: detect → probation passes (age=1 >= probation_age=1) → output
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 1u);
    }

    // Frame 2: detect again → still tracked
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 1u);
    }

    // Frame 3: no detection → track goes to Shadow, not in output
    {
        const auto out = tracker.update({});
        EXPECT_EQ(out.size(), 0u);
    }

    // Frame 4: still no detection → shadow_age=2, still within grace period
    {
        const auto out = tracker.update({});
        EXPECT_EQ(out.size(), 0u);
    }

    // Frame 5: no detection → shadow_age=3 > max_shadow_tracking_age=2 → Removed
    {
        const auto out = tracker.update({});
        EXPECT_EQ(out.size(), 0u);
        // All tracks are gone
        EXPECT_EQ(tracker.getAllTracks().size(), 0u);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Shadow track can be re-matched and returns to Tracked
// ---------------------------------------------------------------------------
TEST(ByteTrack, ShadowReacquire)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.probation_age = 1;
    cfg.max_shadow_tracking_age = 3;
    byte_track::BYTETracker tracker(cfg);

    const auto obj = makeObject(100.f, 100.f, 50.f, 80.f);

    // Establish track
    tracker.update({obj});
    tracker.update({obj});
    size_t tid = 0;
    {
        auto out = tracker.update({obj});
        ASSERT_EQ(out.size(), 1u);
        tid = out[0]->getTrackId();
    }

    // Miss two frames → Shadow
    tracker.update({});
    tracker.update({});

    // Re-detect → back to Tracked with the same track ID
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 1u);
        if (!out.empty()) EXPECT_EQ(out[0]->getTrackId(), tid);
    }
}

// ---------------------------------------------------------------------------
// Test 4: Probation age — model track not output until probation_age frames
// ---------------------------------------------------------------------------
TEST(ByteTrack, ProbationAge)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.probation_age = 3;  // need 3 matches before appearing in output
    cfg.max_shadow_tracking_age = 5;
    byte_track::BYTETracker tracker(cfg);

    const auto obj = makeObject(50.f, 50.f, 40.f, 60.f);

    // Frame 1: age=1 < 3 → not in output
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 0u);
    }
    // Frame 2: age=2 < 3 → not in output
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 0u);
    }
    // Frame 3: age=3 >= 3 → appears in output
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 1u);
    }
    // Frame 4: still tracked
    {
        const auto out = tracker.update({obj});
        EXPECT_EQ(out.size(), 1u);
    }
}

// ---------------------------------------------------------------------------
// Test 5: Blob probation — blob-seeded track requires blob_probation_age hits
// ---------------------------------------------------------------------------
TEST(ByteTrack, BlobProbation)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.probation_age = 1;          // model tracks confirm in 1 frame
    cfg.blob_probation_age = 3;     // blob tracks need 3 hits
    cfg.max_shadow_tracking_age = 5;
    byte_track::BYTETracker tracker(cfg);

    // Model track confirms in 1 frame
    {
        byte_track::BYTETracker t2(cfg);
        const auto obj = makeObject(300.f, 300.f, 50.f, 80.f);
        const auto out1 = t2.update({obj});
        EXPECT_EQ(out1.size(), 1u);  // confirmed at frame 1
    }

    // Blob track: center (200, 200), radius 25 → bbox (175, 175, 50, 50)
    const auto blob = makeBlob(200.f, 200.f, 25.f, 1.0f);

    // Frame 1: blob_hits=1 < 3 → not in output
    {
        const auto out = tracker.update({}, {blob});
        EXPECT_EQ(out.size(), 0u);
    }
    // Frame 2: blob_hits=2 < 3 → not in output
    {
        const auto out = tracker.update({}, {blob});
        EXPECT_EQ(out.size(), 0u);
    }
    // Frame 3: blob_hits=3 >= 3 → appears in output
    {
        const auto out = tracker.update({}, {blob});
        EXPECT_EQ(out.size(), 1u);
    }

    // Now supply a model detection at same location → is_blob_track cleared
    {
        const auto model_obj = makeObject(175.f, 175.f, 50.f, 50.f);
        const auto out = tracker.update({model_obj}, {});
        EXPECT_EQ(out.size(), 1u);
        if (!out.empty()) EXPECT_FALSE(out[0]->isBlobTrack());
    }
}

// ---------------------------------------------------------------------------
// Test 6: Early termination — probationary track with no match is removed fast
// ---------------------------------------------------------------------------
TEST(ByteTrack, EarlyTermination)
{
    byte_track::BYTETrackerConfig cfg;
    cfg.probation_age = 5;
    cfg.early_termination_age = 1;  // remove after 1 shadow frame
    byte_track::BYTETracker tracker(cfg);

    // Frame 1: seed a new track
    {
        const auto out = tracker.update({makeObject(400.f, 400.f, 50.f, 80.f)});
        EXPECT_EQ(out.size(), 0u);  // still in probation
        EXPECT_EQ(tracker.getTentativeTracks().size(), 1u);
    }

    // Frame 2: no detection → shadow_age=1 >= early_termination_age=1 → removed
    {
        const auto out = tracker.update({});
        EXPECT_EQ(out.size(), 0u);
        EXPECT_EQ(tracker.getAllTracks().size(), 0u);
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
