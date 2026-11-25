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
    private readonly HeightProjection _projection;
    private readonly TrackerOptions _options;
    private readonly BallFilter _filter;

    public BallTracker(TableCalibration calibration, EngineConfig engine, TrackerOptions options)
    {
        _projection = new HeightProjection(calibration, options);
        _options = options;
        _filter = new BallFilter(engine);
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

        var p = _projection.ToTable(new Point2d(ball.Center.X, ball.Center.Y), _options.BallHeight);

        var margin = _options.OutsideMargin;
        var table = _projection.Table;
        if (p.X < -margin || p.X > table.Length + margin || p.Y < -margin || p.Y > table.Width + margin)
            return BallObservation.Missing;
        return new BallObservation(p.X, p.Y, ball.Confidence, true);
    }

    /// <summary>Pixel where a ball at this table position is seen; undoes the parallax correction of <see cref="Observe"/>.</summary>
    public Point2d ToImage(double x, double y) => _projection.ToImage(new Point2d(x, y), _options.BallHeight);
}
