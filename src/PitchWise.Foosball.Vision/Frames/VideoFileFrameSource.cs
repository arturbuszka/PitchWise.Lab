using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Frames;

/// <summary>Frames decoded from a video file with OpenCV.</summary>
public sealed class VideoFileFrameSource : IFrameSource
{
    private readonly VideoCapture _capture;

    public VideoFileFrameSource(string path)
    {
        _capture = new VideoCapture(path);
        if (!_capture.IsOpened())
        {
            _capture.Dispose();
            throw new FileNotFoundException($"Could not open video {path}", path);
        }
        Fps = _capture.Fps;
        Width = _capture.FrameWidth;
        Height = _capture.FrameHeight;
    }

    public double Fps { get; }
    public int Width { get; }
    public int Height { get; }

    public IEnumerable<VideoFrame> ReadFrames()
    {
        for (long index = 0; ; index++)
        {
            var image = new Mat();
            if (!_capture.Read(image) || image.Empty())
            {
                image.Dispose();
                yield break;
            }
            // After Read, PosMsec is the presentation time of the frame just decoded.
            var timestamp = TimeSpan.FromMilliseconds(_capture.Get(VideoCaptureProperties.PosMsec));
            yield return new VideoFrame(index, timestamp, image);
        }
    }

    public void Dispose() => _capture.Dispose();
}
