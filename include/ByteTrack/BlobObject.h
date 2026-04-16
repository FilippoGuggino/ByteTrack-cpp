#pragma once

namespace byte_track
{
// Represents a blob detection (e.g. from a LoG blob detector).
// The tracker converts this to a square bounding box centered at (x, y)
// with side length 2 * radius.
struct BlobObject
{
    float x;              // center column (pixels)
    float y;              // center row (pixels)
    float radius;         // detection radius (pixels); bbox = (x-r, y-r, 2r, 2r)
    float response = 1.0f; // detector response magnitude (used as confidence proxy)
    int label = -1;       // optional class label (-1 = unknown)
};
}
