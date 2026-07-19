using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Engine;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Tests;

/// <summary>
/// Covers the progress/cancellation seam on the offline engine with stand-in modules, so the
/// plumbing is tested without ONNX models or a GPU.
/// </summary>
public class TrackingProgressTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"progress-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    /// <summary>Emits one detection per frame, so detection-level modules downstream have work.</summary>
    private sealed class StubDetector(int batchSize) : ImageLevelModule(batchSize)
    {
        public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();
        public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } =
            new Dictionary<string, string[]> { ["detection"] = ["bbox_ltwh"] };
        public override string Level => "image";

        public override object Preprocess(Mat image, DataFrame detections, Series metadata) => metadata.Index;

        public override ModuleOutput Process(IReadOnlyList<object> batch, DataFrame detections,
            IReadOnlyList<Series> metadatas)
        {
            var rows = new List<Series>();
            foreach (var metadata in metadatas)
            {
                var row = new Series(metadata.Index);
                row["image_id"] = metadata.Index;
                row["bbox_ltwh"] = new[] { 1f, 2f, 3f, 4f };
                rows.Add(row);
            }
            return new ModuleOutput(rows);
        }
    }

    private sealed class StubDetectionModule(int batchSize) : DetectionLevelModule(batchSize)
    {
        public override IReadOnlyDictionary<string, string[]> InputColumns { get; } =
            new Dictionary<string, string[]> { ["detection"] = ["bbox_ltwh"] };
        public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } =
            new Dictionary<string, string[]> { ["detection"] = ["score"] };
        public override string Level => "detection";

        public override object Preprocess(Mat image, Series detection, Series metadata) => detection.Index;

        public override IReadOnlyList<Series> Process(IReadOnlyList<object> batch, DataFrame detections,
            IReadOnlyList<Series> metadatas)
        {
            var rows = new List<Series>();
            foreach (var index in batch.Cast<long>())
            {
                var row = new Series(index);
                row["score"] = 1.0f;
                rows.Add(row);
            }
            return rows;
        }
    }

    private sealed class StubVideoModule : VideoLevelModule
    {
        public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>();
        public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>();
        public override string Level => "video";

        public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas) => detections;
    }

    private TrackerState CreateState(int frameCount)
    {
        Directory.CreateDirectory(_root);
        using var frame = new Mat(new Size(16, 16), MatType.CV_8UC3, new Scalar(0, 0, 0));

        var videoMetadatas = new DataFrame();
        var imageMetadatas = new DataFrame();
        var video = videoMetadatas.NewRow();
        video["id"] = 1L;
        video["name"] = "stub";

        for (var i = 0; i < frameCount; i++)
        {
            var path = Path.Combine(_root, $"{i:D4}.jpg");
            Cv2.ImWrite(path, frame);
            var image = imageMetadatas.NewRow();
            image["id"] = image.Index;
            image["video_id"] = 1L;
            image["frame"] = (long)i;
            image["file_path"] = path;
        }

        return new TrackerState(new TrackingSet
        {
            VideoMetadatas = videoMetadatas,
            ImageMetadatas = imageMetadatas,
        });
    }

    [Fact]
    public void Progress_ReportsAtFrameGranularityAcrossEveryModuleLevel()
    {
        var state = CreateState(frameCount: 10);
        var pipeline = new Pipeline(
            [new StubDetector(batchSize: 4), new StubDetectionModule(batchSize: 4), new StubVideoModule()]);
        var ticks = new List<TrackingProgress>();
        // reports on the calling thread, unlike Progress<T>, so the sequence is assertable
        var engine = new OfflineTrackingEngine(pipeline, state)
        {
            Progress = new SynchronousProgress(ticks.Add),
        };

        engine.VideoLoop(1L);

        Assert.NotEmpty(ticks);
        Assert.All(ticks, t => Assert.Equal(3, t.ModuleCount));
        Assert.All(ticks, t => Assert.Equal(1L, t.VideoId));

        // every module reported, in pipeline order, with its own index
        Assert.Equal([0, 1, 2], ticks.Select(t => t.ModuleIndex).Distinct());
        Assert.Equal("StubDetector", ticks.First(t => t.ModuleIndex == 0).ModuleName);
        Assert.Equal("StubVideoModule", ticks.First(t => t.ModuleIndex == 2).ModuleName);

        // the image-level module walked all 10 frames in batches of 4: 0, 4, 8, 10
        var detector = ticks.Where(t => t.ModuleIndex == 0).ToList();
        Assert.All(detector, t => Assert.Equal(10, t.ItemsTotal));
        Assert.Equal([0, 4, 8, 10], detector.Select(t => t.ItemsDone));

        // and each module finished at 100%
        foreach (var index in new[] { 0, 1, 2 })
        {
            var last = ticks.Last(t => t.ModuleIndex == index);
            Assert.Equal(1.0, last.ModuleFraction);
        }
    }

    [Fact]
    public void Cancellation_StopsWithinAModuleRatherThanAtTheNextBoundary()
    {
        var state = CreateState(frameCount: 40);
        var pipeline = new Pipeline([new StubDetector(batchSize: 4), new StubDetectionModule(batchSize: 4)]);
        using var cts = new CancellationTokenSource();
        var ticks = new List<TrackingProgress>();
        var engine = new OfflineTrackingEngine(pipeline, state)
        {
            CancellationToken = cts.Token,
            Progress = new SynchronousProgress(t =>
            {
                ticks.Add(t);
                if (t.ItemsDone >= 8) cts.Cancel();
            }),
        };

        Assert.Throws<OperationCanceledException>(() => engine.VideoLoop(1L));

        // stopped inside the first module, nowhere near the 40-frame boundary
        Assert.All(ticks, t => Assert.Equal(0, t.ModuleIndex));
        Assert.True(ticks.Last().ItemsDone <= 12, $"ran on to {ticks.Last().ItemsDone} frames after cancelling at 8");
    }

    [Fact]
    public void NullProgress_LeavesTheEngineExactlyAsItWas()
    {
        var state = CreateState(frameCount: 6);
        var pipeline = new Pipeline([new StubDetector(batchSize: 4), new StubDetectionModule(batchSize: 4)]);

        // the pre-existing three-argument construction still compiles and runs untouched
        var engine = new OfflineTrackingEngine(pipeline, state);
        Assert.Null(engine.Progress);
        Assert.Equal(CancellationToken.None, engine.CancellationToken);

        var (detections, imagePreds) = engine.VideoLoop(1L);

        Assert.Equal(6, imagePreds.Count);
        Assert.Equal(6, detections.Count);
        Assert.All(detections.Rows, d => Assert.Equal(1.0f, d.Get<float>("score")));
    }

    /// <summary>IProgress that invokes on the calling thread, so tests can assert on order.</summary>
    private sealed class SynchronousProgress(Action<TrackingProgress> handler) : IProgress<TrackingProgress>
    {
        public void Report(TrackingProgress value) => handler(value);
    }
}
