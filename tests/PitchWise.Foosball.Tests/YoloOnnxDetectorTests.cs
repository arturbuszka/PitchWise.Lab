using OpenCvSharp;
using PitchWise.Foosball.Vision.Detection;

namespace PitchWise.Foosball.Tests;

/// <summary>
/// Runs the detector on labelled frames from the held-out video 5 (Data/frames, YOLO labels next
/// to each image). Runs on CPU so the result does not depend on the machine's GPU setup.
/// </summary>
public sealed class YoloOnnxDetectorTests : IDisposable
{
    private const float BallTolerancePx = 10;
    private const int FigureTolerance = 1;

    private readonly YoloOnnxDetector? _detector;

    public YoloOnnxDetectorTests()
    {
        if (FoosballModel.RepoRoot is null || !File.Exists(FoosballModel.Path)) return;
        var config = DetectorOptions.Load(Path.Combine(FoosballModel.RepoRoot, "configs/foosball/detector.json"));
        _detector = new YoloOnnxDetector(config with { ModelPath = FoosballModel.Path, UseGpu = false });
    }

    public void Dispose() => _detector?.Dispose();

    [RequiresModelFact]
    public void FindsTheLabelledBall()
    {
        // free ball, motion-blurred shot, ball partly under a figure, ball next to a hand
        foreach (var name in new[] { "free-0003", "fast-0282", "partial-0243", "hand-0014" })
        {
            var (detections, labels, _) = Run(name);
            var expected = Assert.Single(labels, l => l.Class == 0).Center;
            Assert.True(detections.Ball is not null, $"{name}: no ball detected");
            var distance = Distance(detections.Ball.Center, expected);
            Assert.True(distance <= BallTolerancePx, $"{name}: ball {distance:F1} px from the label");
        }
    }

    [RequiresModelFact]
    public void ReportsNoBallWhenThereIsNone()
    {
        var (detections, _, _) = Run("noball-0089");
        Assert.Null(detections.Ball);
    }

    [RequiresModelFact]
    public void FindsTheLabelledFigures()
    {
        foreach (var name in new[] { "free-0003", "fast-0282", "partial-0243", "noball-0089" })
        {
            var (detections, labels, _) = Run(name);
            var expected = labels.Count(l => l.Class == 1);
            Assert.True(Math.Abs(detections.Figures.Count - expected) <= FigureTolerance,
                $"{name}: {detections.Figures.Count} figures, labelled {expected}");
        }
    }

    private (FrameDetections Detections, List<Label> Labels, Size Size) Run(string name)
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "Data", "frames");
        using var image = Cv2.ImRead(Path.Combine(dir, name + ".jpg"));
        var labels = File.ReadAllLines(Path.Combine(dir, name + ".txt"))
            .Select(line => line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            .Where(p => p.Length == 5)
            .Select(p => new Label(int.Parse(p[0]), new Point2f(
                float.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture) * image.Width,
                float.Parse(p[2], System.Globalization.CultureInfo.InvariantCulture) * image.Height)))
            .ToList();
        return (_detector!.Detect(image), labels, image.Size());
    }

    private static double Distance(Point2f a, Point2f b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    /// <summary>A YOLO label converted to pixels: class and box centre.</summary>
    private sealed record Label(int Class, Point2f Center);
}
