using OpenCvSharp;
using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Workspace;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Tests;

public class PlaybackTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"playback-test-{Guid.NewGuid():N}");

    public PlaybackTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    /// <summary>Frames on disk plus the image metadata an analysis would have recorded.</summary>
    /// <remarks>Written as png, not jpg: the tests identify frames by pixel value, and jpg is lossy.</remarks>
    private DataFrame CreateFrames(int count, int width = 64, int height = 36)
    {
        var imagePreds = new DataFrame();
        for (var i = 0; i < count; i++)
        {
            var path = Path.Combine(_root, $"{i:D4}.png");
            // a distinct shade per frame, so tests can tell which one came back
            using var image = new Mat(new Size(width, height), MatType.CV_8UC3,
                new Scalar(10 + i * 10, 20, 30));
            Cv2.ImWrite(path, image);

            var row = imagePreds.NewRow();
            row["id"] = row.Index;
            row["video_id"] = 1L;
            row["frame"] = (long)i;
            row["file_path"] = path;
        }
        return imagePreds;
    }

    [Fact]
    public void FrameSequenceSource_SeeksExactlyAndInAnyOrder()
    {
        var imagePreds = CreateFrames(5);
        using var source = new FrameSequenceSource(imagePreds, fps: 25);
        using var frame = new Mat();

        Assert.Equal(5, source.FrameCount);
        Assert.Equal(25, source.Fps);
        Assert.Equal(new Size(64, 36), source.FrameSize);
        Assert.True(source.SeekIsExact);

        // random access: frame 4 costs the same as frame 0 and lands exactly
        Assert.True(source.TryRead(4, frame));
        Assert.Equal(50, frame.At<Vec3b>(0, 0).Item0);   // 10 + 4*10

        Assert.True(source.TryRead(1, frame));
        Assert.Equal(20, frame.At<Vec3b>(0, 0).Item0);

        Assert.False(source.TryRead(-1, frame));
        Assert.False(source.TryRead(5, frame));
    }

    [Fact]
    public void FrameSequenceSource_MetadataCarriesTheImageIdDetectionsAreKeyedBy()
    {
        var imagePreds = CreateFrames(3);
        using var source = new FrameSequenceSource(imagePreds, fps: 25);

        // the overlay joins detections on image_id, so this link is the whole contract
        var metadata = source.Metadata(2);
        Assert.NotNull(metadata);
        Assert.Equal(2L, metadata.Get<long>("frame"));
        Assert.Equal(metadata.Index, metadata.Get<long>("id"));
        Assert.Null(source.Metadata(99));
    }

    [Fact]
    public void FrameSequenceSource_SaysSoWhenTheFramesAreGone()
    {
        var imagePreds = CreateFrames(2);
        File.Delete(Path.Combine(_root, "0000.png"));

        var ex = Assert.Throws<FileNotFoundException>(() => new FrameSequenceSource(imagePreds, 25));
        Assert.Contains("deleted", ex.Message);
    }

    [Fact]
    public void FrameSourceFactory_PrefersTheFramesAndFallsBackToTheVideo()
    {
        var imagePreds = CreateFrames(2);
        var manifest = new AnalysisManifest
        {
            JobId = "j", DisplayName = "clip", SourceKind = SourceKind.FrameSequence,
            SourcePath = _root, CreatedUtc = DateTimeOffset.UtcNow, Status = JobStatus.Completed,
            Fps = 25,
        };
        var factory = new FrameSourceFactory();

        using (var source = factory.Create(manifest, imagePreds))
        {
            Assert.IsType<FrameSequenceSource>(source);
            Assert.True(source.SeekIsExact);
        }

        // frames gone and the source is not a video: nothing to fall back to, so say why
        foreach (var file in Directory.GetFiles(_root, "*.png")) File.Delete(file);
        var ex = Assert.Throws<InvalidOperationException>(() => factory.Create(manifest, imagePreds));
        Assert.Contains("frames", ex.Message);
    }

    [Fact]
    public void OverlayRenderer_IndexesDetectionsByImageIdRatherThanRescanningThem()
    {
        var detections = new DataFrame();
        foreach (var (imageId, trackId) in new[] { (10L, 1L), (10L, 2L), (11L, 3L) })
        {
            var row = detections.NewRow();
            row["image_id"] = imageId;
            row["track_id"] = trackId;
        }
        using var renderer = new OverlayRenderer(detections);

        Assert.Equal([1L, 2L], renderer.DetectionsOf(10).Select(d => d.Get<long>("track_id")));
        Assert.Equal([3L], renderer.DetectionsOf(11).Select(d => d.Get<long>("track_id")));
        Assert.Empty(renderer.DetectionsOf(99));
        Assert.False(renderer.HasGroundTruth);
    }

    [Fact]
    public void OverlayRenderer_DrawsOnlyWhatTheOptionsAskFor()
    {
        var detections = new DataFrame();
        var detection = detections.NewRow();
        detection["image_id"] = 0L;
        detection["track_id"] = 7L;
        detection["bbox_ltwh"] = new[] { 10f, 10f, 20f, 40f };
        detection["role"] = "player";
        detection["team"] = "left";
        detection["bbox_pitch"] = new BboxPitch(-10, 5, -9, 5, -9.5, 5);

        var imagePreds = new DataFrame();
        var metadata = new Series(0L);
        metadata["id"] = 0L;
        imagePreds.Upsert(metadata);

        using var renderer = new OverlayRenderer(detections);
        using var blank = new Mat(new Size(320, 180), MatType.CV_8UC3, Scalar.Black);
        using var drawn = blank.Clone();

        renderer.Draw(drawn, metadata, new OverlayOptions(
            Pitch: false, Ellipses: false, Radar: false, GroundTruthRadar: false));
        Assert.Equal(0, Cv2.CountNonZero(Difference(blank, drawn)));   // everything off: untouched

        renderer.Draw(drawn, metadata, new OverlayOptions(
            Pitch: false, Ellipses: true, Radar: false, GroundTruthRadar: false));
        Assert.True(Cv2.CountNonZero(Difference(blank, drawn)) > 0, "the ellipse should have drawn");
    }

    [Fact]
    public void OverlayRenderer_ReportsGroundTruthOnlyWhenThereIsSome()
    {
        var detections = new DataFrame();
        detections.NewRow()["image_id"] = 0L;

        using var without = new OverlayRenderer(detections, new DataFrame());
        Assert.False(without.HasGroundTruth);

        var gt = new DataFrame();
        gt.NewRow()["image_id"] = 0L;
        using var with = new OverlayRenderer(detections, gt);
        Assert.True(with.HasGroundTruth);
    }

    [Fact]
    public void PlaybackLoop_SeeksStepsAndPlaysToTheEndWithoutWrapping()
    {
        var imagePreds = CreateFrames(5);
        using var source = new FrameSequenceSource(imagePreds, fps: 25);
        using var overlay = new OverlayRenderer(new DataFrame());
        var seen = new List<int>();
        var gate = new object();

        using var loop = new PlaybackLoop(source, overlay, new InstantClock(), frame =>
        {
            lock (gate) seen.Add(frame.Index);
        });

        loop.SeekTo(3);
        WaitUntil(() => Last(seen, gate) == 3);
        Assert.Equal(3, loop.CurrentIndex);

        loop.Step(-1);
        WaitUntil(() => Last(seen, gate) == 2);
        Assert.False(loop.IsPlaying, "stepping should leave playback paused");

        var beforePlay = Count(seen, gate);
        loop.Play();
        WaitUntil(() => !loop.IsPlaying);   // runs to the end and stops there

        Assert.Equal(4, loop.CurrentIndex);
        // resumed at 3 and stopped at the last frame rather than wrapping round to 0
        lock (gate) Assert.Equal([3, 4], seen.Skip(beforePlay));
    }

    [Fact]
    public void PlaybackLoop_SeekingWhilePausedRedrawsWithoutStartingPlayback()
    {
        var imagePreds = CreateFrames(4);
        using var source = new FrameSequenceSource(imagePreds, fps: 25);
        using var overlay = new OverlayRenderer(new DataFrame());
        var count = 0;

        using var loop = new PlaybackLoop(source, overlay, new InstantClock(),
            _ => Interlocked.Increment(ref count));

        loop.SeekTo(1);
        WaitUntil(() => Volatile.Read(ref count) == 1);

        // toggling the overlay while paused must redraw the current frame, not advance
        loop.SetOverlay(new OverlayOptions(Radar: false));
        WaitUntil(() => Volatile.Read(ref count) == 2);

        Assert.Equal(1, loop.CurrentIndex);
        Assert.False(loop.IsPlaying);
    }

    /// <summary>Paces nothing, so the loop runs as fast as it can decode.</summary>
    private sealed class InstantClock : IFrameClock
    {
        public void WaitNextFrame(double fps, double rate, CancellationToken ct) { }
    }

    private static int Last(List<int> seen, object gate)
    {
        lock (gate) return seen.Count > 0 ? seen[^1] : -1;
    }

    private static int Count(List<int> seen, object gate)
    {
        lock (gate) return seen.Count;
    }

    private static void WaitUntil(Func<bool> condition)
    {
        for (var i = 0; i < 200 && !condition(); i++) Thread.Sleep(5);
        Assert.True(condition(), "the loop never reached the expected state");
    }

    private static Mat Difference(Mat a, Mat b)
    {
        var diff = new Mat();
        Cv2.Absdiff(a, b, diff);
        Cv2.CvtColor(diff, diff, ColorConversionCodes.BGR2GRAY);
        return diff;
    }
}
