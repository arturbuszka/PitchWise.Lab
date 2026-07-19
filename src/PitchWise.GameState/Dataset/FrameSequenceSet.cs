using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Dataset;

/// <summary>
/// Builds a TrackingSet from a directory of numbered images, without copying or re-encoding:
/// file_path points straight at the originals. The counterpart of <see cref="ExternalVideo.Load"/>
/// for material that is already frames on disk.
/// </summary>
/// <remarks>
/// SoccerNetGameState.LoadVideo needs a Labels-GameState.json beside the frames; this loader
/// needs nothing but the images, which is what a plain capture or an already-extracted video
/// looks like. Frame rate cannot be recovered from loose images, so the caller supplies it.
/// </remarks>
public static class FrameSequenceSet
{
    private static readonly string[] Extensions = [".jpg", ".jpeg", ".png", ".bmp"];

    /// <param name="framesDir">Directory of images, ordinal filename order = frame order.</param>
    /// <param name="fps">Frame rate of the source material; loose images do not record one.</param>
    /// <param name="frameStride">Keep every n-th image (1 = all).</param>
    /// <param name="maxFrames">Stop after this many kept frames (0 = no limit).</param>
    public static TrackingSet Load(string framesDir, double fps = 25, int frameStride = 1,
        int maxFrames = 0)
    {
        if (!Directory.Exists(framesDir))
            throw new DirectoryNotFoundException($"frames directory not found: {framesDir}");
        if (frameStride < 1)
            throw new ArgumentOutOfRangeException(nameof(frameStride), frameStride, "must be >= 1");

        var files = Directory.EnumerateFiles(framesDir)
            .Where(f => Extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.Ordinal)
            .Where((_, i) => i % frameStride == 0)
            .ToList();
        if (maxFrames > 0 && files.Count > maxFrames)
            files = files.Take(maxFrames).ToList();
        if (files.Count == 0)
            throw new FileNotFoundException($"no images in {framesDir}");

        // resolution is only discoverable by decoding one; the pipeline needs it up front
        using var first = Cv2.ImRead(files[0]);
        if (first.Empty())
            throw new InvalidDataException($"could not decode {files[0]}");

        var videoMetadatas = new DataFrame();
        var imageMetadatas = new DataFrame();

        var video = videoMetadatas.NewRow();
        video["id"] = 1L;
        video["name"] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(framesDir)));
        video["frame_rate"] = (long)Math.Round(fps);
        video["im_width"] = (long)first.Width;
        video["im_height"] = (long)first.Height;
        video["nframes"] = (long)files.Count;

        for (var i = 0; i < files.Count; i++)
        {
            var image = imageMetadatas.NewRow();
            image["id"] = image.Index;
            image["video_id"] = 1L;
            image["frame"] = (long)i;
            image["file_path"] = files[i];
            image["nframes"] = (long)files.Count;
        }

        return new TrackingSet { VideoMetadatas = videoMetadatas, ImageMetadatas = imageMetadatas };
    }
}
