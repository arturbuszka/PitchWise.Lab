using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>
/// Frames read from the images on disk that the analysis itself worked from.
/// </summary>
/// <remarks>
/// The default source, and the reason playback is built around frames rather than video: seeking
/// is exact and costs the same for frame 700 as for frame 1, whereas seeking an H.264 file is
/// approximate. Since the overlay is keyed by image_id, an approximate seek would draw one
/// frame's players over another frame's pixels. Decoding a 1080p jpg is ~3-8 ms, inside a 40 ms
/// frame budget.
/// </remarks>
public sealed class FrameSequenceSource : IFrameSource
{
    private readonly Series[] _frames;

    /// <param name="imagePreds">Image metadata of the analysis, carrying file_path and frame.</param>
    /// <param name="fps">Frame rate recorded in the manifest.</param>
    public FrameSequenceSource(DataFrame imagePreds, double fps)
    {
        _frames = imagePreds.Rows
            .Where(r => r.Has("file_path"))
            .OrderBy(r => r.GetOrDefault<long>("frame"))
            .ToArray();
        Fps = fps > 0 ? fps : 25;

        if (_frames.Length == 0)
            throw new InvalidOperationException("the analysis recorded no frames");

        using var first = Cv2.ImRead(_frames[0].Get<string>("file_path"), ImreadModes.Color);
        if (first.Empty())
            throw new FileNotFoundException(
                $"could not read {_frames[0].Get<string>("file_path")}; the frames may have been deleted");
        FrameSize = first.Size();
    }

    public int FrameCount => _frames.Length;

    public double Fps { get; }

    public Size FrameSize { get; }

    public bool SeekIsExact => true;

    public bool TryRead(int index, Mat destination)
    {
        if (index < 0 || index >= _frames.Length) return false;

        using var image = Cv2.ImRead(_frames[index].Get<string>("file_path"), ImreadModes.Color);
        if (image.Empty()) return false;
        image.CopyTo(destination);
        return true;
    }

    public Series? Metadata(int index)
        => index >= 0 && index < _frames.Length ? _frames[index] : null;

    public void Dispose()
    {
        // nothing held open: each frame is read and released on demand
    }
}
