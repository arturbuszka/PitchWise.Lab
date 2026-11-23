using OpenCvSharp;
using PitchWise.Engine;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;
using PitchWise.Foosball.Vision.Tracking;

namespace PitchWise.Foosball.Tests;

/// <summary>Uses the shipped configs, so retuning them cannot quietly break tracking.</summary>
public class BallTrackerTests
{
    private const double FrameSeconds = 1 / 240.0;
    private const double PixelsPerMetre = 2500;
    private const double CentreX = 0.2925, CentreY = 0.171;

    private static readonly TableModel Table = TableModel.Load(Config("table.json"));
    private static readonly EngineConfig Engine = FoosballEngineConfig.Load(Config("engine.json"), Table);
    private static readonly TrackerOptions Options = TrackerOptions.Load(Config("tracker.json"));

    private readonly BallTracker _tracker = new(Calibration(), Engine, Options);

    [Fact]
    public void ShippedEngineConfigDescribesTheTable()
    {
        Assert.Equal(Table.Length, Engine.PitchLength);
        Assert.Equal(Table.Width, Engine.PitchWidth);
        Assert.InRange(Engine.BallMeasurementNoise, 0.0001, 0.02);
    }

    [Fact]
    public void MisspeltEngineKeyIsRejected()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, """{ "maxBalSpeed": 1 }""");
            Assert.Throws<System.Text.Json.JsonException>(() => FoosballEngineConfig.Load(path, Table));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void BallBelowTheCameraIsNotShifted()
    {
        var ball = _tracker.Observe(SeenAt(CentreX, CentreY));

        Assert.Equal(CentreX, ball.X, 1e-4);
        Assert.Equal(CentreY, ball.Y, 1e-4);
    }

    [Fact]
    public void BallAtTheGoalIsPulledInwardsByParallax()
    {
        // 0.2925 m from the camera axis, ball centre 17 mm up, camera at 615 mm: 8.1 mm.
        Assert.Equal(0.0081, _tracker.Observe(SeenAt(0, CentreY)).X, 2e-4);
        Assert.Equal(Table.Length - 0.0081, _tracker.Observe(SeenAt(Table.Length, CentreY)).X, 2e-4);
    }

    [Fact]
    public void BallFarOffTheTableIsMissing()
    {
        Assert.False(_tracker.Observe(SeenAt(-0.08, CentreY)).Detected);
        Assert.False(_tracker.Observe(SeenAt(CentreX, Table.Width + 0.08)).Detected);
        Assert.False(_tracker.Observe(new FrameDetections(null, [])).Detected);
    }

    [Fact]
    public void FollowsAStraightRollDespiteNoise()
    {
        var random = new Random(1);
        BallState state = default;
        for (var i = 0; i <= 60; i++)
            state = Step(i, 0.1 + i * FrameSeconds + (random.NextDouble() - 0.5) * 0.002).State;

        Assert.Equal(0.1 + 60 * FrameSeconds, state.X, 0.003);
        Assert.Equal(1.0, state.Vx, 0.25);
        Assert.False(state.Interpolated);
    }

    [Fact]
    public void PredictsThroughAShortOcclusion()
    {
        for (var i = 0; i <= 60; i++) Step(i, 0.1 + i * FrameSeconds);
        BallState state = default;
        for (var i = 61; i <= 72; i++) // 50 ms
        {
            state = Step(i, null).State;
            Assert.True(state.Interpolated);
            Assert.True(state.Confidence > 0);
        }

        // The ball really moved 50 mm; staying put would be 50 mm off.
        Assert.InRange(Math.Abs(0.1 + 72 * FrameSeconds - state.X), 0, 0.02);
    }

    [Fact]
    public void RejectsASingleImpossibleJump()
    {
        for (var i = 0; i <= 30; i++) Step(i, 0.1);

        var state = Step(31, 0.5).State;

        Assert.True(state.Interpolated);
        Assert.Equal(0.1, state.X, 0.005);
    }

    [Fact]
    public void FindsTheBallElsewhereAfterALongOcclusion()
    {
        for (var i = 0; i <= 30; i++) Step(i, 0.1);
        var lost = 31 + (int)(Engine.MaxCoastSeconds / FrameSeconds) + 5;
        BallState state = default;
        for (var i = 31; i < lost; i++) state = Step(i, null).State;
        Assert.Equal(0, state.Confidence);

        state = Step(lost, 0.5).State;

        Assert.False(state.Interpolated);
        Assert.Equal(0.5, state.X, 0.001);
    }

    private static string Config(string name) => Path.Combine(AppContext.BaseDirectory, name);

    private static TableCalibration Calibration()
    {
        // Straight-down camera, 100 px border: pixels are metres times a scale.
        float right = (float)(100 + Table.Length * PixelsPerMetre), bottom = (float)(100 + Table.Width * PixelsPerMetre);
        return TableCalibration.FromCorners(
            [new(100, 100), new(right, 100), new(right, bottom), new(100, bottom)], new Size(1920, 1080), Table);
    }

    /// <summary>A detection whose centre falls on the given mat point (before parallax).</summary>
    private static FrameDetections SeenAt(double x, double y) => new(
        new Detection(new Rect2f((float)(100 + x * PixelsPerMetre - 20), (float)(100 + y * PixelsPerMetre - 20), 40, 40), 0.9f), []);

    /// <summary>One frame with the ball truly at x on the centre line, or hidden when null.</summary>
    private BallTrack Step(int frame, double? x)
    {
        var detections = x is { } trueX
            ? SeenAt(CentreX + (trueX - CentreX) * Options.CameraHeight / (Options.CameraHeight - Options.BallHeight), CentreY)
            : new FrameDetections(null, []);
        return _tracker.Step(detections, TimeSpan.FromSeconds(frame * FrameSeconds));
    }
}
