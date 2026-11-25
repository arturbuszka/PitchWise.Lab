using OpenCvSharp;
using PitchWise.Engine;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;
using PitchWise.Foosball.Vision.Tracking;

namespace PitchWise.Foosball.Tests;

public class TrackOverlayTests
{
    private static readonly TableModel Table = TableModel.Load(Path.Combine(AppContext.BaseDirectory, "table.json"));

    private readonly BallTracker _tracker = new(
        TableCalibration.FromCorners([new(100, 100), new(700, 100), new(700, 450), new(100, 450)], new Size(800, 600), Table),
        new EngineConfig(), new TrackerOptions(0.017, 0.615, 0.03));

    [Fact]
    public void ToImageUndoesTheParallaxCorrection()
    {
        var ball = _tracker.Observe(new FrameDetections(new Detection(new Rect2f(110, 140, 20, 20), 0.9f), []));

        var pixel = _tracker.ToImage(ball.X, ball.Y);

        Assert.Equal(120, pixel.X, 0.01);
        Assert.Equal(150, pixel.Y, 0.01);
    }

    [Fact]
    public void MeasuredAndPredictedPartsHaveDifferentColours()
    {
        var overlay = new TrackOverlay(_tracker, 0.5);
        Draw(overlay, 200, 0, predicted: false);
        Draw(overlay, 300, 100, predicted: false);
        using var image = Draw(overlay, 400, 200, predicted: true);

        var measured = image.At<Vec3b>(300, 250);
        var predicted = image.At<Vec3b>(300, 350);
        Assert.NotEqual(default, measured);
        Assert.NotEqual(default, predicted);
        Assert.NotEqual(measured, predicted);
    }

    [Fact]
    public void OldTrailAndLostBallAreNotDrawn()
    {
        var overlay = new TrackOverlay(_tracker, 0.5);
        Draw(overlay, 200, 0, predicted: false);
        using var later = Draw(overlay, 400, 600, predicted: false);
        Assert.Equal(default, later.At<Vec3b>(300, 300));

        using var lost = Draw(overlay, 400, 610, predicted: true, confidence: 0);
        Assert.Equal(0, Cv2.CountNonZero(lost.CvtColor(ColorConversionCodes.BGR2GRAY)));

        using var found = Draw(overlay, 600, 620, predicted: false);
        Assert.Equal(default, found.At<Vec3b>(300, 500)); // no line from before it was lost
    }

    private Mat Draw(TrackOverlay overlay, double pixelX, double ms, bool predicted, double confidence = 0.9)
    {
        var image = new Mat(600, 800, MatType.CV_8UC3, Scalar.Black);
        var at = _tracker.Observe(new FrameDetections(
            new Detection(new Rect2f((float)pixelX - 10, 290, 20, 20), 0.9f), []));
        overlay.Draw(image, new BallTrack(at, new BallState(at.X, at.Y, 0, 0, 0, predicted, confidence)), TimeSpan.FromMilliseconds(ms));
        return image;
    }
}
