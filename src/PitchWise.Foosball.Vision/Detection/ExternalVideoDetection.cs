using OpenCvSharp;

namespace PitchWise.Foosball.Vision.Detection;

public struct VideoMetadata
{
    public long Id;
    public string Name;
    public double FrameRate;
    public long ImWidth;
    public long ImHeight;
    public long NFrames;
}
/// <summary>
/// Detects objects in an arbitrary video
/// file and returns a VideoMetadata struct.
/// </summary>
public static class ExternalVideoDetection
{
    /// <param name="frameStride">Keep every n-th frame (1 = all).</param>
    /// <param name="maxFrames">Stop after this many kept frames (0 = no limit).</param>
    /// <param name="startSeconds">Skip into the video before extracting.</param>
    public static VideoMetadata Load(string videoPath, string videoDir, int frameStride = 1,
        int maxFrames = 0, double startSeconds = 0)
    {

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
            throw new FileNotFoundException($"Could not open video {videoPath}");
        Directory.CreateDirectory(videoDir);
        if (startSeconds > 0)
            capture.Set(VideoCaptureProperties.PosMsec, startSeconds * 1000);

        var videoMetadata = new VideoMetadata
        {
            Id = 1L,
            Name = Path.GetFileNameWithoutExtension(videoPath),
            FrameRate = capture.Fps / frameStride,
            ImWidth = (long)capture.FrameWidth,
            ImHeight = (long)capture.FrameHeight
        };

        using var frame = new Mat();
        var filePath = Path.Combine(videoDir, $"output_{videoMetadata.Name}_{DateTime.Now:yyyyMMdd_HHmmss}.mp4");
        using VideoWriter writer = new(filePath, FourCC.MP4S, videoMetadata.FrameRate, new OpenCvSharp.Size(videoMetadata.ImWidth, videoMetadata.ImHeight));
        if (!writer.IsOpened())
            throw new FileNotFoundException($"Could not open video writer {filePath}");

        long kept = 0, index = 0;
        while (capture.Read(frame) && !frame.Empty())
        {
            if (index++ % frameStride != 0) continue;
            writer.Write(frame);
            kept++;
            if (maxFrames > 0 && kept >= maxFrames) break;
        }

        if (kept == 0)
            throw new InvalidOperationException(
                $"Opened {videoPath} but decoded no frames — the bundled OpenCV probably cannot decode its codec (re-encode to H.264)");

        videoMetadata.NFrames = kept;
        return videoMetadata;
    }
}
