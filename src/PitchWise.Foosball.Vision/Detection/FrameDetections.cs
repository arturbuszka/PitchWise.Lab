using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Detection;

/// <summary>One detected object, in pixels of the original frame.</summary>
public sealed record Detection(Rect2f Box, float Confidence)
{
    public Point2f Center => new(Box.X + Box.Width / 2, Box.Y + Box.Height / 2);
}

/// <summary>
/// Everything the detector found on one frame. There is one ball on the table, so at most one
/// ball comes back: the most confident one. Taking the best box instead of running NMS also drops
/// the second box the model sometimes puts on a half-hidden ball (see docs/foosball/eval.md).
/// </summary>
public sealed record FrameDetections(Detection? Ball, IReadOnlyList<Detection> Figures);
