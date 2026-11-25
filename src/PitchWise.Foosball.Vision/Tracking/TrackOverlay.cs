using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Tracking;

/// <summary>
/// Draws the recent ball path onto review frames. Stateful, one instance per video. Predicted stretches
/// have their own colour, since they are the ones to check by eye: the detector saw nothing there.
/// </summary>
public sealed class TrackOverlay(BallTracker tracker, double trailSeconds)
{
    private static readonly Scalar MeasuredColor = new(0, 255, 0); // green
    private static readonly Scalar PredictedColor = new(0, 128, 255); // orange

    private readonly Queue<(TimeSpan Time, Point Pixel, bool Predicted)> _trail = new();

    public void Draw(Mat image, BallTrack track, TimeSpan timestamp)
    {
        // A lost ball has no position; clearing also keeps the trail from jumping to where it is found again.
        if (track.State.Confidence <= 0)
        {
            _trail.Clear();
            return;
        }

        var p = tracker.ToImage(track.State.X, track.State.Y);
        _trail.Enqueue((timestamp, new Point((int)Math.Round(p.X), (int)Math.Round(p.Y)), track.State.Interpolated));
        while ((timestamp - _trail.Peek().Time).TotalSeconds > trailSeconds) _trail.Dequeue();

        Point? previous = null;
        foreach (var (_, pixel, predicted) in _trail)
        {
            if (previous is { } from) Cv2.Line(image, from, pixel, ColorOf(predicted), 2);
            previous = pixel;
        }
        Cv2.Circle(image, previous!.Value, 10, ColorOf(track.State.Interpolated), 2);
    }

    private static Scalar ColorOf(bool predicted) => predicted ? PredictedColor : MeasuredColor;
}
