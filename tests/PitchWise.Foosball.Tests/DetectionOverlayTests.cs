using OpenCvSharp;
using PitchWise.Foosball.Vision.Detection;

namespace PitchWise.Foosball.Tests;

public class DetectionOverlayTests
{
    [Fact]
    public void DrawsTheBallBoxWhereItWasDetected()
    {
        using var image = new Mat(100, 100, MatType.CV_8UC3, Scalar.Black);
        var ball = new Detection(new Rect2f(40, 40, 20, 20), 0.9f);

        DetectionOverlay.Draw(image, new FrameDetections(ball, []));

        Assert.NotEqual(default(Vec3b), image.At<Vec3b>(40, 50)); // top edge
        Assert.NotEqual(default(Vec3b), image.At<Vec3b>(50, 60)); // right edge
        Assert.Equal(default(Vec3b), image.At<Vec3b>(50, 50)); // inside stays untouched
    }

    [Fact]
    public void LeavesTheFrameUntouchedWithoutDetections()
    {
        using var image = new Mat(100, 100, MatType.CV_8UC3, Scalar.Black);

        DetectionOverlay.Draw(image, new FrameDetections(null, []));

        Assert.Equal(0, Cv2.CountNonZero(image.CvtColor(ColorConversionCodes.BGR2GRAY)));
    }
}
