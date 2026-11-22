using System.Diagnostics;
using OpenCvSharp;
using PitchWise.Foosball.Vision.Detection;
using PitchWise.Foosball.Vision.Frames;

/// <summary>The <c>foosball</c> commands: offline analysis of table recordings.</summary>
static class FoosballCommands
{
    public const string DefaultDetectorConfig = "configs/foosball/detector.json";

    /// <summary>
    /// Runs the detector over a recording and writes it back with boxes drawn. Prints the share of
    /// processed frames with a ball, the quick health check of the model on new footage.
    /// </summary>
    /// <param name="stride">Process every Nth frame; the output plays at fps / N so its speed stays real.</param>
    /// <param name="maxFrames">Stop after this many processed frames; 0 means the whole file.</param>
    public static int Detect(string input, string output, string detectorConfig, int stride, int maxFrames)
    {
        if (stride < 1)
        {
            Console.Error.WriteLine("--stride must be at least 1");
            return 1;
        }
        var options = DetectorOptions.Load(detectorConfig);
        using var source = new VideoFileFrameSource(input);
        using var detector = new YoloOnnxDetector(options);
        Console.WriteLine($"{input}: {source.Width}x{source.Height} @ {source.Fps:F2} fps, " +
                          $"detector on {detector.Provider}, every {stride}. frame");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        // MPEG-4 caps the timebase denominator at 65535, so 239.493 fps cannot be stored exactly.
        var outputFps = Math.Round(source.Fps / stride, 2);
        using var writer = new VideoWriter(output, FourCC.FromString("mp4v"), outputFps, new Size(source.Width, source.Height));
        if (!writer.IsOpened())
        {
            Console.Error.WriteLine($"cannot write {output}");
            return 1;
        }

        int totalFrames;
        using (var probe = new VideoCapture(input))
            totalFrames = (int)probe.Get(VideoCaptureProperties.FrameCount);
        var planned = maxFrames > 0 ? Math.Min(maxFrames, totalFrames / stride) : totalFrames / stride;
        int processed = 0, withBall = 0, lastPercent = -1;
        var detectTime = new Stopwatch();
        foreach (var frame in source.ReadFrames())
        {
            using (frame)
            {
                if (frame.Index % stride != 0) continue;

                detectTime.Start();
                var detections = detector.Detect(frame.Image);
                detectTime.Stop();

                if (detections.Ball is not null) withBall++;
                DetectionOverlay.Draw(frame.Image, detections);
                writer.Write(frame.Image);
                processed++;

                var percent = planned > 0 ? processed * 100 / planned : 0;
                if (percent / 5 != lastPercent / 5)
                {
                    lastPercent = percent;
                    Console.WriteLine($"  {percent}% ({processed}/{planned})");
                }
                if (maxFrames > 0 && processed >= maxFrames) break;
            }
        }

        var share = processed > 0 ? 100.0 * withBall / processed : 0;
        var msPerFrame = processed > 0 ? detectTime.Elapsed.TotalMilliseconds / processed : 0;
        Console.WriteLine($"frames: {processed}, with ball: {withBall} ({share:F1}%), " +
                          $"detector {msPerFrame:F1} ms/frame on {detector.Provider}");
        Console.WriteLine($"written: {output}");
        return 0;
    }
}
