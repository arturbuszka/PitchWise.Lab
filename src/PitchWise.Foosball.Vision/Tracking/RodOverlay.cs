using OpenCvSharp;
using PitchWise.Foosball.Vision.Calibration;

namespace PitchWise.Foosball.Vision.Tracking;

/// <summary>
/// Draws tracked rods and their figures onto review frames. Held positions have their own colour;
/// a rod whose position is unknown gets only its line.
/// </summary>
public sealed class RodOverlay(HeightProjection projection, double rodHeight)
{
    private static readonly Scalar MeasuredColor = new(255, 255, 0); // cyan
    private static readonly Scalar HeldColor = new(0, 128, 255); // orange
    private static readonly Scalar UnknownColor = new(128, 128, 128);

    public void Draw(Mat image, IReadOnlyList<RodState> rods)
    {
        foreach (var state in rods)
        {
            var color = state.Confidence <= 0 ? UnknownColor : state.Measured ? MeasuredColor : HeldColor;
            Cv2.Line(image, Pixel(state.Rod.X, 0), Pixel(state.Rod.X, projection.Table.Width), color, 1);
            if (state.Confidence <= 0) continue;
            foreach (var y in state.FigureYs)
                Cv2.Circle(image, Pixel(state.Rod.X, y), 9, color, 2);
        }
    }

    private Point Pixel(double x, double y)
    {
        var p = projection.ToImage(new Point2d(x, y), rodHeight);
        return new Point((int)Math.Round(p.X), (int)Math.Round(p.Y));
    }
}
