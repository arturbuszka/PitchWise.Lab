using System.Diagnostics;
using System.Globalization;
using OpenCvSharp;
using PitchWise.Foosball;
using PitchWise.Foosball.Vision.Calibration;
using PitchWise.Foosball.Vision.Detection;
using PitchWise.Foosball.Vision.Frames;
using PitchWise.Foosball.Vision.Tracking;

/// <summary>The <c>foosball</c> commands: offline analysis of table recordings.</summary>
static class FoosballCommands
{
    public const string DefaultDetectorConfig = "configs/foosball/detector.json";
    public const string DefaultTableConfig = "configs/foosball/table.json";
    public const string DefaultEngineConfig = "configs/foosball/engine.json";
    public const string DefaultTrackerConfig = "configs/foosball/tracker.json";

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
        using var source = new VideoFileFrameSource(input);
        using var detector = new YoloOnnxDetector(DetectorOptions.Load(detectorConfig));
        int withBall = 0;
        var processed = ProcessVideo(source, detector, input, output, stride, maxFrames, (frame, detections) =>
        {
            if (detections.Ball is not null) withBall++;
            DetectionOverlay.Draw(frame.Image, detections);
        });
        if (processed < 0) return 1;
        Console.WriteLine($"with ball: {Percent(withBall, processed)}");
        return 0;
    }

    /// <summary>
    /// Tracks the ball and the rods through a recording and writes it back with both drawn. Prints how many
    /// frames were measured, predicted and lost, which is what tracker tuning is judged by.
    /// </summary>
    public static int Track(string input, string output, string? calibration, string tableConfig, string engineConfig,
        string trackerConfig, string detectorConfig, int stride, int maxFrames)
    {
        if (calibration is null)
        {
            Console.Error.WriteLine("--calib <calib.json> is required, see foosball calibrate");
            return 1;
        }
        if (stride < 1)
        {
            Console.Error.WriteLine("--stride must be at least 1");
            return 1;
        }
        var table = TableModel.Load(tableConfig);
        var options = TrackerOptions.Load(trackerConfig);
        using var source = new VideoFileFrameSource(input);
        var tableCalibration = TableCalibration.Load(calibration, new Size(source.Width, source.Height), table);
        var tracker = new BallTracker(tableCalibration, FoosballEngineConfig.Load(engineConfig, table), options);
        var overlay = new TrackOverlay(tracker, options.TrailSeconds);
        var rodTracker = new RodTracker(tableCalibration, options);
        var rodOverlay = new RodOverlay(new HeightProjection(tableCalibration, options), options.RodHeight);
        using var detector = new YoloOnnxDetector(DetectorOptions.Load(detectorConfig));

        int measured = 0, predicted = 0, lost = 0;
        var rodMeasured = new int[table.Rods.Count];
        var processed = ProcessVideo(source, detector, input, output, stride, maxFrames, (frame, detections) =>
        {
            var track = tracker.Step(detections, frame.Timestamp);
            if (track.State.Confidence <= 0) lost++;
            else if (track.State.Interpolated) predicted++;
            else measured++;
            var rods = rodTracker.Step(detections, frame.Timestamp);
            for (var i = 0; i < rods.Count; i++)
                if (rods[i].Measured) rodMeasured[i]++;
            rodOverlay.Draw(frame.Image, rods);
            overlay.Draw(frame.Image, track, frame.Timestamp);
        });
        if (processed < 0) return 1;
        for (var i = 0; i < table.Rods.Count; i++)
            Console.WriteLine($"rod {table.Rods[i].Owner} {table.Rods[i].Role}: measured {Percent(rodMeasured[i], processed)}");
        Console.WriteLine($"ball measured: {Percent(measured, processed)}, predicted: {Percent(predicted, processed)}, " +
                          $"lost: {Percent(lost, processed)}");
        return 0;
    }

    private static string Percent(int count, int total) =>
        $"{count} ({(total > 0 ? 100.0 * count / total : 0):F1}%)";

    /// <summary>Detects on every Nth frame, lets <paramref name="draw"/> mark it, and writes the result.</summary>
    /// <returns>Number of processed frames, or -1 when the output cannot be written.</returns>
    private static int ProcessVideo(VideoFileFrameSource source, YoloOnnxDetector detector, string input, string output,
        int stride, int maxFrames, Action<VideoFrame, FrameDetections> draw)
    {
        Console.WriteLine($"{input}: {source.Width}x{source.Height} @ {source.Fps:F2} fps, " +
                          $"detector on {detector.Provider}, every {stride}. frame");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        // MPEG-4 caps the timebase denominator at 65535, so 239.493 fps cannot be stored exactly.
        var outputFps = Math.Round(source.Fps / stride, 2);
        using var writer = new VideoWriter(output, FourCC.FromString("mp4v"), outputFps, new Size(source.Width, source.Height));
        if (!writer.IsOpened())
        {
            Console.Error.WriteLine($"cannot write {output}");
            return -1;
        }

        int totalFrames;
        using (var probe = new VideoCapture(input))
            totalFrames = (int)probe.Get(VideoCaptureProperties.FrameCount);
        var planned = maxFrames > 0 ? Math.Min(maxFrames, totalFrames / stride) : totalFrames / stride;
        int processed = 0, lastPercent = -1;
        var detectTime = new Stopwatch();
        var total = Stopwatch.StartNew();
        foreach (var frame in source.ReadFrames())
        {
            using (frame)
            {
                if (frame.Index % stride != 0) continue;

                detectTime.Start();
                var detections = detector.Detect(frame.Image);
                detectTime.Stop();

                draw(frame, detections);
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

        var msPerFrame = processed > 0 ? detectTime.Elapsed.TotalMilliseconds / processed : 0;
        Console.WriteLine($"frames: {processed}, detector {msPerFrame:F1} ms/frame, " +
                          $"all {total.Elapsed.TotalMilliseconds / Math.Max(1, processed):F1} ms/frame");
        Console.WriteLine($"written: {output}");
        return processed;
    }

    /// <summary>
    /// Builds a calibration from four hand-picked playfield corners and writes it next to a preview
    /// (<c>calib.jpg</c>): the table model drawn back onto the first frame, so a wrong corner shows
    /// as rods and goals off their real place.
    /// </summary>
    /// <param name="corners">"x,y" pixels in the order (0,0), (L,0), (L,W), (0,W) of the table model.</param>
    public static int Calibrate(string video, string calibPath, string tableConfig, string[] corners)
    {
        var points = new Point2f[corners.Length];
        for (var i = 0; i < corners.Length; i++)
        {
            var parts = corners[i].Split(',');
            if (parts.Length != 2
                || !float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var x)
                || !float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var y))
            {
                Console.Error.WriteLine($"corner '{corners[i]}' is not x,y");
                return 1;
            }
            points[i] = new Point2f(x, y);
        }

        var table = TableModel.Load(tableConfig);
        using var source = new VideoFileFrameSource(video);
        using var frame = source.ReadFrames().FirstOrDefault();
        if (frame is null)
        {
            Console.Error.WriteLine($"{video} has no frames");
            return 1;
        }

        var calibration = TableCalibration.FromCorners(points, new Size(source.Width, source.Height), table);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(calibPath))!);
        calibration.Save(calibPath);

        DrawTableModel(frame.Image, calibration);
        var preview = Path.ChangeExtension(calibPath, ".jpg");
        Cv2.ImWrite(preview, frame.Image);
        Console.WriteLine($"written: {calibPath}, preview: {preview}");
        return 0;
    }

    /// <summary>Outline, goal mouths and rods of the table model, projected into the image.</summary>
    private static void DrawTableModel(Mat image, TableCalibration calibration)
    {
        var t = calibration.Table;
        var outline = new Scalar(0, 255, 255);
        var goal = new Scalar(0, 0, 255);
        var rod = new Scalar(255, 255, 255);
        Point Px(double x, double y)
        {
            var p = calibration.ToImage(new Point2d(x, y));
            return new Point((int)Math.Round(p.X), (int)Math.Round(p.Y));
        }

        Cv2.Polylines(image, [[Px(0, 0), Px(t.Length, 0), Px(t.Length, t.Width), Px(0, t.Width)]], true, outline, 2);
        var g0 = (t.Width - t.GoalWidth) / 2;
        var g1 = (t.Width + t.GoalWidth) / 2;
        Cv2.Line(image, Px(0, g0), Px(0, g1), goal, 5);
        Cv2.Line(image, Px(t.Length, g0), Px(t.Length, g1), goal, 5);
        foreach (var r in t.Rods)
        {
            Cv2.Line(image, Px(r.X, 0), Px(r.X, t.Width), rod, 1);
            // Figures at rest position: centred across the table.
            var first = (t.Width - (r.FigureCount - 1) * r.FigureSpacing) / 2;
            for (var i = 0; i < r.FigureCount; i++)
                Cv2.Circle(image, Px(r.X, first + i * r.FigureSpacing), 4, rod, -1);
        }
    }
}
