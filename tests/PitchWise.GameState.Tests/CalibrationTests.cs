using PitchWise.GameState.Calibration;

namespace PitchWise.GameState.Tests;

public class CalibrationTests
{
    /// <summary>A plausible broadcast-camera world->image homography used as ground truth.</summary>
    private static readonly double[,] TrueH =
    {
        { 14.0, -3.0, 960.0 },
        { 1.2, 6.5, 700.0 },
        { 0.0002, 0.009, 1.0 },
    };

    private static (double X, double Y) Project(double[,] h, double x, double y)
        => FrameByFrameCalib.Transform(h, x, y);

    private static Dictionary<int, Keypoint> SyntheticKeypoints(
        int imageWidth, int imageHeight, Func<int, bool>? include = null, double noise = 0)
    {
        var rng = new Random(42);
        var keypoints = new Dictionary<int, Keypoint>();
        for (var id = 1; id <= 57; id++)
        {
            if (PitchGeometry.CrossbarKeypoints.Contains(id)) continue;
            if (include is not null && !include(id)) continue;
            var (xw, yw) = PitchGeometry.WorldCoords(id);
            var (xi, yi) = Project(TrueH, xw, yw);
            xi += (rng.NextDouble() - 0.5) * 2 * noise;
            yi += (rng.NextDouble() - 0.5) * 2 * noise;
            if (xi < 0 || xi >= imageWidth || yi < 0 || yi >= imageHeight) continue;
            // module receives normalized coordinates
            keypoints[id] = new Keypoint(xi / imageWidth, yi / imageHeight, 0.9);
        }
        return keypoints;
    }

    [Fact]
    public void Homography_RecoveredFromCleanKeypoints_ProjectsAccurately()
    {
        const int w = 1920, h = 1080;
        var calib = new FrameByFrameCalib(w, h, denormalize: true);
        calib.Update(SyntheticKeypoints(w, h));

        var hInv = calib.GetHomographyFromGroundPlane(useRansac: 50, inverse: true);
        Assert.NotNull(hInv);

        // Any image point projected to the pitch must match the true inverse mapping.
        foreach (var (xw, yw) in new[] { (0.0, 0.0), (-20.0, 10.0), (30.0, -25.0) })
        {
            var (xi, yi) = Project(TrueH, xw, yw);
            var (xp, yp) = Project(hInv!, xi, yi);
            Assert.True(Math.Abs(xp - xw) < 0.05, $"x: {xp} vs {xw}");
            Assert.True(Math.Abs(yp - yw) < 0.05, $"y: {yp} vs {yw}");
        }
    }

    [Fact]
    public void Homography_WithNoisyOutlier_IsRejectedByRansac()
    {
        const int w = 1920, h = 1080;
        var keypoints = SyntheticKeypoints(w, h, noise: 1.5);
        // corrupt one keypoint heavily — RANSAC (threshold 50 px) should discard it
        var firstKey = keypoints.Keys.First();
        keypoints[firstKey] = new Keypoint(0.99, 0.01, 0.9);

        var calib = new FrameByFrameCalib(w, h, denormalize: true);
        calib.Update(keypoints);
        var hInv = calib.GetHomographyFromGroundPlane(useRansac: 50, inverse: true);
        Assert.NotNull(hInv);

        var (xi, yi) = Project(TrueH, 0, 0);
        var (xp, yp) = Project(hInv!, xi, yi);
        Assert.True(Math.Abs(xp) < 0.5 && Math.Abs(yp) < 0.5, $"center projected to ({xp}, {yp})");
    }

    [Fact]
    public void Homography_TooFewKeypoints_ReturnsNull()
    {
        var calib = new FrameByFrameCalib(1920, 1080, denormalize: true);
        calib.Update(SyntheticKeypoints(1920, 1080, include: id => id is 1 or 2 or 3));
        Assert.Null(calib.GetHomographyFromGroundPlane(useRansac: 50, inverse: true));
    }

    [Fact]
    public void GetBboxPitch_ProjectsBottomPointsOntoPitch()
    {
        const int w = 1920, h = 1080;
        var calib = new FrameByFrameCalib(w, h, denormalize: true);
        calib.Update(SyntheticKeypoints(w, h));
        var hInv = calib.GetHomographyFromGroundPlane(useRansac: 50, inverse: true)!;

        // A player standing at pitch position (10, 5): bbox bottom-middle at its image projection.
        var (xi, yi) = Project(TrueH, 10, 5);
        var bbox = new[] { (float)(xi - 20), (float)(yi - 100), (float)(xi + 20), (float)yi };
        var pitch = NbjwCalib.GetBboxPitch(hInv, bbox);

        Assert.NotNull(pitch);
        Assert.True(Math.Abs(pitch!.XBottomMiddle - 10) < 0.1, $"x: {pitch.XBottomMiddle}");
        Assert.True(Math.Abs(pitch.YBottomMiddle - 5) < 0.1, $"y: {pitch.YBottomMiddle}");
        Assert.True(pitch.XBottomLeft < pitch.XBottomRight);
    }

    [Fact]
    public void KeypointCompleter_ReconstructsCornerFromLineIntersection()
    {
        // Side line top: horizontal at y=100; side line left: vertical at x=200 -> corner kp 1 at (200, 100).
        var lines = new Dictionary<int, LineSegment>
        {
            [Array.IndexOf(PitchGeometry.LinesList, "Side line top") + 1] =
                new(new Keypoint(300, 100, 1), new Keypoint(900, 100, 1)),
            [Array.IndexOf(PitchGeometry.LinesList, "Side line left") + 1] =
                new(new Keypoint(200, 300, 1), new Keypoint(201, 500, 1)),
        };
        var complete = KeypointCompleter.Complete(new Dictionary<int, Keypoint>(), lines, 960, 540);

        Assert.True(complete.ContainsKey(1));
        Assert.True(Math.Abs(complete[1].X - 200) < 3, $"x: {complete[1].X}");
        Assert.True(Math.Abs(complete[1].Y - 100) < 3, $"y: {complete[1].Y}");
    }

    [Fact]
    public void HomographyInvert_RoundTripsPoints()
    {
        var inv = FrameByFrameCalib.Invert(TrueH);
        Assert.NotNull(inv);
        var (xi, yi) = Project(TrueH, -15, 20);
        var (xw, yw) = Project(inv!, xi, yi);
        Assert.True(Math.Abs(xw + 15) < 1e-6 && Math.Abs(yw - 20) < 1e-6);
    }
}
