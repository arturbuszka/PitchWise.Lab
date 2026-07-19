using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>
/// Frames decoded from the source video, for jobs whose extracted frames have been deleted.
/// </summary>
/// <remarks>
/// A documented degradation, not an equal alternative: <see cref="SeekIsExact"/> is false because
/// OpenCV seeks long-GOP H.264 to the nearest keyframe and decodes forward, so a requested frame
/// can come back off by one or several. The overlay is keyed by image_id, so the players drawn
/// may belong to a neighbouring frame. Sequential playback is unaffected — only seeking.
/// </remarks>
public sealed class VideoFileSource : IFrameSource
{
    private readonly VideoCapture _capture;
    private readonly Series[] _frames;
    private int _position = -1;

    public VideoFileSource(string videoPath, DataFrame imagePreds, double fps)
    {
        _capture = new VideoCapture(videoPath);
        if (!_capture.IsOpened())
            throw new FileNotFoundException($"could not open {videoPath}");

        _frames = imagePreds.Rows.OrderBy(r => r.GetOrDefault<long>("frame")).ToArray();
        Fps = fps > 0 ? fps : _capture.Fps;
        FrameSize = new Size(_capture.FrameWidth, _capture.FrameHeight);
    }

    public int FrameCount => _frames.Length;

    public double Fps { get; }

    public Size FrameSize { get; }

    public bool SeekIsExact => false;

    public bool TryRead(int index, Mat destination)
    {
        if (index < 0 || index >= _frames.Length) return false;

        // playing forward is the common case and needs no seek at all, which also keeps it exact
        if (index != _position + 1)
            _capture.Set(VideoCaptureProperties.PosFrames, index);

        if (!_capture.Read(destination) || destination.Empty())
        {
            _position = -1;
            return false;
        }
        _position = index;
        return true;
    }

    public Series? Metadata(int index)
        => index >= 0 && index < _frames.Length ? _frames[index] : null;

    public void Dispose() => _capture.Dispose();
}
