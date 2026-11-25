using OpenCvSharp;
using PitchWise.Engine;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;

namespace PitchWise.Foosball.Vision.Tracking;

/// <summary>
/// One tracked frame. The raw observation is kept next to the filtered state because
/// <c>WorldStateBuilder</c> (3.8) runs its own filter: feeding it the state would filter twice.
/// </summary>
public sealed record BallTrack(BallObservation Observation, BallState State);

/// <summary>
/// Turns ball boxes into a continuous trajectory in table metres. Stateful: feed frames in order.
/// </summary>
public sealed class BallTracker
{
    private readonly TableCalibration _calibration;
    private readonly TrackerOptions _options;
    private readonly BallFilter _filter;
    private readonly Point2d _belowCamera;

    public BallTracker(TableCalibration calibration, EngineConfig engine, TrackerOptions options)
    {
        _calibration = calibration;
        _options = options;
        _filter = new BallFilter(engine);
        _belowCamera = new Point2d(
            options.CameraX ?? calibration.Table.Length / 2,
            options.CameraY ?? calibration.Table.Width / 2);
    }

    /// <param name="timestamp">Frame time from the stream, so uneven frame spacing does not become false speed.</param>
    public BallTrack Step(FrameDetections detections, TimeSpan timestamp)
    {
        var observation = Observe(detections);
        return new BallTrack(observation, _filter.Step(observation, timestamp.TotalSeconds));
    }

    /// <summary>The detected ball in table metres; missing when there is none or it lies off the table.</summary>
    public BallObservation Observe(FrameDetections detections)
    {
        if (detections.Ball is not { } ball) return BallObservation.Missing;

        var seen = _calibration.ToTable(new Point2d(ball.Center.X, ball.Center.Y));
        // The homography puts the ball where its line of sight meets the mat, beyond the ball itself;
        // moving back towards the point below the camera by the height ratio gives the real position.
        var x = _belowCamera.X + (seen.X - _belowCamera.X) * ParallaxScale;
        var y = _belowCamera.Y + (seen.Y - _belowCamera.Y) * ParallaxScale;

        var margin = _options.OutsideMargin;
        var table = _calibration.Table;
        if (x < -margin || x > table.Length + margin || y < -margin || y > table.Width + margin)
            return BallObservation.Missing;
        return new BallObservation(x, y, ball.Confidence, true);
    }

    /// <summary>Pixel where a ball at this table position is seen; undoes the parallax correction of <see cref="Observe"/>.</summary>
    public Point2d ToImage(double x, double y) => _calibration.ToImage(new Point2d(
        _belowCamera.X + (x - _belowCamera.X) / ParallaxScale,
        _belowCamera.Y + (y - _belowCamera.Y) / ParallaxScale));

    private double ParallaxScale => (_options.CameraHeight - _options.BallHeight) / _options.CameraHeight;
}
