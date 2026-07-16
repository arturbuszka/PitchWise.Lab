using OpenCvSharp;

namespace PitchWise.GameState.Calibration;

/// <summary>
/// Per-frame camera calibration, ground-plane path:
/// computes the homography between the pitch plane (world meters, origin at center)
/// and the image from the detected keypoints, with the same two-stage RANSAC scheme
/// (outlier pre-filter, then final estimation on the surviving correspondences).
/// </summary>
public sealed partial class FrameByFrameCalib
{
    private readonly int _imageWidth;
    private readonly int _imageHeight;
    private readonly bool _denormalize;
    private Dictionary<int, Keypoint> _keypoints = new();

    public FrameByFrameCalib(int imageWidth = 960, int imageHeight = 540, bool denormalize = false)
    {
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;
        _denormalize = denormalize;
    }

    public void Update(IReadOnlyDictionary<int, Keypoint> keypoints)
    {
        _keypoints = keypoints.ToDictionary(
            kv => kv.Key,
            kv => _denormalize
                ? kv.Value with { X = kv.Value.X * _imageWidth, Y = kv.Value.Y * _imageHeight }
                : kv.Value);
    }

    /// <summary>
    /// get_homography_from_ground_plane: world (pitch plane) -> image homography, or its
    /// inverse (image -> pitch) when <paramref name="inverse"/> is set. Returns null when
    /// fewer than 4 usable correspondences exist or the estimate is degenerate.
    /// </summary>
    public double[,]? GetHomographyFromGroundPlane(double useRansac = 5.0, bool inverse = false)
    {
        // ground-plane subset: everything except the goal crossbar keypoints (z != 0)
        var correspondences = _keypoints
            .Where(kv => !PitchGeometry.CrossbarKeypoints.Contains(kv.Key))
            .Select(kv =>
            {
                var (xw, yw) = PitchGeometry.WorldCoords(kv.Key);
                return (World: new Point2d(xw, yw), Image: new Point2d(kv.Value.X, kv.Value.Y));
            })
            .ToList();

        if (useRansac > 0 && !IsDegenerate(correspondences))
        {
            // Pre-filter stage of get_per_plane_correspondences: keep RANSAC inliers only.
            using var mask = new Mat();
            using var pre = Cv2.FindHomography(
                correspondences.Select(c => c.World), correspondences.Select(c => c.Image),
                HomographyMethods.Ransac, useRansac, mask);
            if (!pre.Empty())
            {
                mask.GetArray(out byte[] status);
                correspondences = correspondences.Where((_, i) => status[i] == 1).ToList();
            }
        }

        if (IsDegenerate(correspondences))
            return null;

        using var h = Cv2.FindHomography(
            correspondences.Select(c => c.World), correspondences.Select(c => c.Image),
            HomographyMethods.Ransac, useRansac);
        if (h.Empty())
            return null;

        h.GetArray(out double[] flat);
        var homography = new double[3, 3];
        for (var i = 0; i < 9; i++) homography[i / 3, i % 3] = flat[i];

        if (!inverse)
            return homography;

        return Invert(homography);
    }

    /// <summary>Inverts a homography and normalizes it so H[2,2] == 1; null when singular.</summary>
    public static double[,]? Invert(double[,] h)
    {
        var det =
            h[0, 0] * (h[1, 1] * h[2, 2] - h[1, 2] * h[2, 1]) -
            h[0, 1] * (h[1, 0] * h[2, 2] - h[1, 2] * h[2, 0]) +
            h[0, 2] * (h[1, 0] * h[2, 1] - h[1, 1] * h[2, 0]);
        if (Math.Abs(det) < 1e-12)
            return null;

        var inv = new double[3, 3];
        inv[0, 0] = (h[1, 1] * h[2, 2] - h[1, 2] * h[2, 1]) / det;
        inv[0, 1] = (h[0, 2] * h[2, 1] - h[0, 1] * h[2, 2]) / det;
        inv[0, 2] = (h[0, 1] * h[1, 2] - h[0, 2] * h[1, 1]) / det;
        inv[1, 0] = (h[1, 2] * h[2, 0] - h[1, 0] * h[2, 2]) / det;
        inv[1, 1] = (h[0, 0] * h[2, 2] - h[0, 2] * h[2, 0]) / det;
        inv[1, 2] = (h[0, 2] * h[1, 0] - h[0, 0] * h[1, 2]) / det;
        inv[2, 0] = (h[1, 0] * h[2, 1] - h[1, 1] * h[2, 0]) / det;
        inv[2, 1] = (h[0, 1] * h[2, 0] - h[0, 0] * h[2, 1]) / det;
        inv[2, 2] = (h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]) / det;

        var norm = inv[2, 2];
        if (Math.Abs(norm) < 1e-12)
            return null;
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                inv[i, j] /= norm;
        return inv;
    }

    /// <summary>Applies a homography to a 2D point (project through the plane).</summary>
    public static (double X, double Y) Transform(double[,] h, double x, double y)
    {
        var w = h[2, 0] * x + h[2, 1] * y + h[2, 2];
        return ((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w,
                (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w);
    }

    private static bool IsDegenerate(List<(Point2d World, Point2d Image)> correspondences)
        => correspondences.Count < 4
           || correspondences.All(c => c.World.X == correspondences[0].World.X)
           || correspondences.All(c => c.World.Y == correspondences[0].World.Y);
}
