using SnGameState.Calibration;

namespace SnGameState.Tests;

public class CameraParametersTests
{
    private const int W = 1920, H = 1080;

    // A realistic main broadcast camera: behind the side line, elevated, tilted down.
    private const double TruePanDeg = 8.0, TrueTiltDeg = 72.0, TrueRollDeg = 0.4;
    private const double TrueFocal = 1200.0;
    private static readonly double[] TruePosition = { 2.0, 45.0, -18.0 };

    /// <summary>Projects a world point with the pinhole model matching the port's conventions
    /// (rotation = orientation^T so that RotationMatrixToPanTiltRoll recovers the angles).</summary>
    private static (double X, double Y) ProjectCamera(double xw, double yw, double zw = 0)
    {
        var rotation = Transpose(FrameByFrameCalib.PanTiltRollToOrientation(
            TruePanDeg * Math.PI / 180, TrueTiltDeg * Math.PI / 180, TrueRollDeg * Math.PI / 180));
        var d = new[] { xw - TruePosition[0], yw - TruePosition[1], zw - TruePosition[2] };
        var cam = new double[3];
        for (var i = 0; i < 3; i++)
            for (var k = 0; k < 3; k++)
                cam[i] += rotation[i, k] * d[k];
        return (TrueFocal * cam[0] / cam[2] + W / 2.0, TrueFocal * cam[1] / cam[2] + H / 2.0);
    }

    private static Dictionary<int, Keypoint> SyntheticKeypoints()
    {
        var keypoints = new Dictionary<int, Keypoint>();
        for (var id = 1; id <= 57; id++)
        {
            var (xw, yw) = PitchGeometry.WorldCoords(id);
            var zw = PitchGeometry.CrossbarKeypoints.Contains(id) ? -2.44 : 0.0;
            var (xi, yi) = ProjectCamera(xw, yw, zw);
            if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
            keypoints[id] = new Keypoint(xi / W, yi / H, 0.9);
        }
        return keypoints;
    }

    [Fact]
    public void GetCamParams_RecoversCameraAnglesAndPosition()
    {
        var calib = new FrameByFrameCalib(W, H, denormalize: true);
        calib.Update(SyntheticKeypoints());

        var result = calib.GetCamParams("full", useRansac: 0);
        Assert.NotNull(result);
        var (p, repErr) = result.Value;

        Assert.True(repErr < 3, $"reprojection error {repErr}");
        Assert.True(Math.Abs(p.PanDegrees - TruePanDeg) < 1.5, $"pan {p.PanDegrees}");
        Assert.True(Math.Abs(p.TiltDegrees - TrueTiltDeg) < 1.5, $"tilt {p.TiltDegrees}");
        Assert.True(Math.Abs(p.RollDegrees - TrueRollDeg) < 1.5, $"roll {p.RollDegrees}");
        Assert.True(Math.Abs(p.PositionMeters[1] - TruePosition[1]) < 5, $"pos y {p.PositionMeters[1]}");
        Assert.True(Math.Abs(p.PositionMeters[2] - TruePosition[2]) < 5, $"pos z {p.PositionMeters[2]}");
        Assert.True(Math.Abs(p.XFocalLength - TrueFocal) / TrueFocal < 0.1, $"focal {p.XFocalLength}");
    }

    [Fact]
    public void HeuristicVoting_ReturnsBestCalibration()
    {
        var calib = new FrameByFrameCalib(W, H, denormalize: true);
        calib.Update(SyntheticKeypoints());

        var parameters = calib.HeuristicVoting();
        Assert.NotNull(parameters);
        Assert.True(Math.Abs(parameters!.TiltDegrees - TrueTiltDeg) < 1.5, $"tilt {parameters.TiltDegrees}");
    }

    [Fact]
    public void HeuristicVoting_NoKeypoints_ReturnsNull()
    {
        var calib = new FrameByFrameCalib(W, H, denormalize: true);
        calib.Update(new Dictionary<int, Keypoint>());
        Assert.Null(calib.HeuristicVoting());
    }

    [Fact]
    public void PanTiltRoll_RoundTripsThroughDecomposition()
    {
        var (pan, tilt, roll) = (0.3, 1.2, -0.02);
        var rotation = Transpose(FrameByFrameCalib.PanTiltRollToOrientation(pan, tilt, roll));
        var (p, t, r) = FrameByFrameCalib.RotationMatrixToPanTiltRoll(rotation);
        Assert.True(Math.Abs(p - pan) < 1e-9 && Math.Abs(t - tilt) < 1e-9 && Math.Abs(r - roll) < 1e-9,
            $"got ({p}, {t}, {r})");
    }

    private static double[,] Transpose(double[,] a)
    {
        var result = new double[3, 3];
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                result[i, j] = a[j, i];
        return result;
    }
}
