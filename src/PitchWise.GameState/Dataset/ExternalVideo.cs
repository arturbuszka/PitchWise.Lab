using OpenCvSharp;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.GameState.Dataset;

/// <summary>
/// Turns an arbitrary video
/// file into a TrackingSet by extracting frames to a working directory (the offline
/// engine and the visualizer both address frames by file path).
/// </summary>
public static class ExternalVideo
{
    /// <param name="frameStride">Keep every n-th frame (1 = all).</param>
    /// <param name="maxFrames">Stop after this many kept frames (0 = no limit).</param>
    /// <param name="startSeconds">Skip into the video before extracting.</param>
    public static TrackingSet Load(string videoPath, string framesDir, int frameStride = 1,
        int maxFrames = 0, double startSeconds = 0)
    {
        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
            throw new FileNotFoundException($"Could not open video {videoPath}");
        Directory.CreateDirectory(framesDir);
        if (startSeconds > 0)
            capture.Set(VideoCaptureProperties.PosMsec, startSeconds * 1000);

        var videoMetadatas = new DataFrame();
        var imageMetadatas = new DataFrame();

        var video = videoMetadatas.NewRow();
        video["id"] = 1L;
        video["name"] = Path.GetFileNameWithoutExtension(videoPath);
        video["frame_rate"] = (long)Math.Round(capture.Fps);
        video["im_width"] = (long)capture.FrameWidth;
        video["im_height"] = (long)capture.FrameHeight;

        var startIndex = (long)capture.Get(VideoCaptureProperties.PosFrames);
        using var frame = new Mat();
        long kept = 0, index = 0;
        while (capture.Read(frame) && !frame.Empty())
        {
            if (index++ % frameStride != 0) continue;
            // named by source frame number so different start offsets don't collide in the cache
            var filePath = Path.Combine(framesDir, $"{startIndex + index:D8}.jpg");
            if (!File.Exists(filePath))
                Cv2.ImWrite(filePath, frame);

            var image = imageMetadatas.NewRow();
            image["id"] = image.Index;
            image["video_id"] = 1L;
            image["frame"] = kept;
            image["file_path"] = filePath;
            kept++;
            if (maxFrames > 0 && kept >= maxFrames) break;
        }

        if (kept == 0)
            throw new InvalidOperationException(
                $"Opened {videoPath} but decoded no frames — the bundled OpenCV probably cannot decode its codec (re-encode to H.264)");

        video["nframes"] = kept;
        return new TrackingSet { VideoMetadatas = videoMetadatas, ImageMetadatas = imageMetadatas };
    }
}
