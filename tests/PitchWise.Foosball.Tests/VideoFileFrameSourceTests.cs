using OpenCvSharp;
using PitchWise.Foosball.Vision.Frames;

namespace PitchWise.Foosball.Tests;

/// <summary>Reads a clip the test writes itself, so it needs no recordings on disk.</summary>
public sealed class VideoFileFrameSourceTests : IDisposable
{
    private const int Frames = 30;
    private const double Fps = 30;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"pw-frames-{Guid.NewGuid():N}.avi");

    public VideoFileFrameSourceTests()
    {
        // MJPG in AVI is built into OpenCV, so the test does not depend on the bundled ffmpeg codecs.
        using var writer = new VideoWriter(_path, FourCC.MJPG, Fps, new Size(64, 48));
        Assert.True(writer.IsOpened());
        for (var i = 0; i < Frames; i++)
        {
            // Brightness encodes the frame number, so skipped or repeated frames are detectable.
            using var frame = new Mat(48, 64, MatType.CV_8UC3, Scalar.All(i * 8));
            writer.Write(frame);
        }
    }

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void ReportsStreamParameters()
    {
        using var source = new VideoFileFrameSource(_path);
        Assert.Equal(Fps, source.Fps, 0.01);
        Assert.Equal(64, source.Width);
        Assert.Equal(48, source.Height);
    }

    [Fact]
    public void ReadsEveryFrameInOrderWithTimestamps()
    {
        using var source = new VideoFileFrameSource(_path);
        var count = 0;
        foreach (var frame in source.ReadFrames())
        {
            using (frame)
            {
                Assert.Equal(count, frame.Index);
                Assert.Equal(count * 8, frame.Image.Mean().Val0, 3);   // JPEG is lossy, hence the tolerance
                Assert.Equal(count / Fps, frame.Timestamp.TotalSeconds, 0.002);
            }
            count++;
        }
        Assert.Equal(Frames, count);
    }

    [Fact]
    public void MissingFileThrows()
    {
        Assert.Throws<FileNotFoundException>(() => new VideoFileFrameSource("/nonexistent/clip.mp4"));
    }
}
