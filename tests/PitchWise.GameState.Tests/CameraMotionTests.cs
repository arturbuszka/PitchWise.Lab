using OpenCvSharp;
using PitchWise.GameState.Track;

namespace PitchWise.GameState.Tests;

public class CameraMotionTests
{
    /// <summary>Two 1920x1080 crops of one smooth random texture: the content moves by (dx, dy) pixels.</summary>
    private static (Mat First, Mat Second) Pan(int dx, int dy)
    {
        using var noise = new Mat(1300, 2200, MatType.CV_8UC3);
        Cv2.Randu(noise, Scalar.All(0), Scalar.All(255));
        using var texture = new Mat();
        Cv2.GaussianBlur(noise, texture, new Size(0, 0), 12);
        Cv2.Normalize(texture, texture, 0, 255, NormTypes.MinMax);
        var first = texture[new Rect(100, 100, 1920, 1080)].Clone();
        var second = texture[new Rect(100 - dx, 100 - dy, 1920, 1080)].Clone();
        return (first, second);
    }

    [Theory]
    [InlineData(CameraMotionMode.Ecc)]
    [InlineData(CameraMotionMode.Flow)]
    public void RecoversCameraPan(CameraMotionMode mode)
    {
        var (first, second) = Pan(-24, 9);
        using var cameraMotion = new CameraMotion(mode);
        Assert.Null(cameraMotion.Next(first, Array.Empty<float[]>()));
        var m = cameraMotion.Next(second, Array.Empty<float[]>());

        Assert.NotNull(m);
        Assert.Equal(-24, m[0, 2], 1.5);
        Assert.Equal(9, m[1, 2], 1.5);
        Assert.Equal(1, m[0, 0], 0.01);
        Assert.Equal(1, m[1, 1], 0.01);
        first.Dispose();
        second.Dispose();
    }

    [Fact]
    public void None_ReturnsNoMotion()
    {
        var (first, second) = Pan(-24, 9);
        using var cameraMotion = new CameraMotion(CameraMotionMode.None);
        Assert.Null(cameraMotion.Next(first, Array.Empty<float[]>()));
        Assert.Null(cameraMotion.Next(second, Array.Empty<float[]>()));
        first.Dispose();
        second.Dispose();
    }
}
