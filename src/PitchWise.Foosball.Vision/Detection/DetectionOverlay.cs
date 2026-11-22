using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Detection;

/// <summary>Draws detections onto a frame for review videos.</summary>
public static class DetectionOverlay
{
    private static readonly Scalar BallColor = new(255, 0, 255); // magenta: stands out on the green field
    private static readonly Scalar FigureColor = new(255, 255, 0); // cyan

    public static void Draw(Mat image, FrameDetections detections)
    {
        foreach (var figure in detections.Figures)
            Cv2.Rectangle(image, ToRect(figure.Box), FigureColor, 1);

        if (detections.Ball is { } ball)
        {
            var rect = ToRect(ball.Box);
            Cv2.Rectangle(image, rect, BallColor, 3);
            Cv2.PutText(image, $"{ball.Confidence:F2}", new Point(rect.X, Math.Max(15, rect.Y - 6)),
                HersheyFonts.HersheySimplex, 0.6, BallColor, 2);
        }
    }

    private static Rect ToRect(Rect2f box) =>
        new((int)Math.Round(box.X), (int)Math.Round(box.Y), (int)Math.Round(box.Width), (int)Math.Round(box.Height));
}
