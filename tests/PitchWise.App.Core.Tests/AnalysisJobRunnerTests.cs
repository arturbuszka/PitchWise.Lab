using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Workspace;
using PitchWise.GameState.Calibration;

namespace PitchWise.App.Core.Tests;

/// <summary>
/// Drives the real pipeline over a handful of real frames — the proof that a full analysis runs
/// with no UI at all.
/// </summary>
/// <remarks>
/// These require the repo's dataset and ONNX models, both of which are present in this checkout,
/// so their absence is asserted rather than skipped past: a test that quietly passes when it
/// tested nothing is worse than no test. Six frames with the OCR-free CPU config keeps the whole
/// thing to a few seconds.
/// </remarks>
public class AnalysisJobRunnerTests : IDisposable
{
    private const string FramesDir = @"data\SoccerNetGS\valid\SNGS-021\img1";
    private const string ModelsDir = "pretrained_models";
    private const string ConfigPath = @"configs\cpu-noocr.yaml";

    private readonly string _root = Path.Combine(Path.GetTempPath(), $"runner-test-{Guid.NewGuid():N}");
    private readonly string? _repo = FindRepoRoot();

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    /// <summary>Walks up from the test binary to the repo, which holds the data and models.</summary>
    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "PitchWise.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    /// <summary>
    /// Creates the job then runs it, mirroring what the queue does: the queue owns job creation
    /// so it can show the job — and leave a tombstone — before the runner ever sees it.
    /// </summary>
    private static Task<AnalysisResult> Run(WorkspaceStore store, AnalysisRequest request,
        IProgress<AnalysisProgress> progress, CancellationToken ct = default)
    {
        var (paths, _) = store.CreateJob(request);
        return new AnalysisJobRunner(store).RunAsync(paths, request, progress, ct);
    }

    /// <summary>Fails loudly rather than skipping: these assets are part of the checkout.</summary>
    private void RequireAssets()
    {
        Assert.True(_repo is not null, "could not locate the repo root from the test binary");
        Assert.True(Directory.Exists(Path.Combine(_repo!, FramesDir)), $"missing frames: {FramesDir}");
        Assert.True(File.Exists(Path.Combine(_repo!, ModelsDir, "yolo11m.onnx")),
            $"missing detector model in {ModelsDir}");
    }

    [Fact]
    public async Task RunAsync_ProducesEveryArtifactAJobFolderPromises()
    {
        RequireAssets();

        var store = new WorkspaceStore(_root);
        var updates = new List<AnalysisProgress>();

        var result = await Run(store, new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Path.Combine(_repo!, FramesDir),
            ConfigPath = Path.Combine(_repo!, ConfigPath),
            ModelsDir = Path.Combine(_repo!, ModelsDir),
            MaxFrames = 6,
        }, new SynchronousProgress(updates.Add), CancellationToken.None);

        var paths = store.PathsFor(result.JobDirectory);
        var manifest = store.ReadManifest(result.JobDirectory);

        Assert.Equal(JobStatus.Completed, manifest.Status);
        Assert.Equal("SNGS-021", manifest.DisplayName);
        Assert.Equal(6, manifest.FrameCount);
        Assert.Equal(1920, manifest.ImWidth);
        Assert.NotNull(manifest.CompletedUtc);

        // which modules actually ran is decided by `enabled: auto`, so the manifest records it
        Assert.Contains("YoloOnnxDetector", manifest.PipelineDescription);

        // every artifact the manifest claims exists on disk
        Assert.True(File.Exists(paths.Stats), "stats.json");
        Assert.True(File.Exists(paths.WorldStateJsonl), "world_state.jsonl");
        Assert.True(File.Exists(paths.Detections), "detections.json");
        Assert.True(File.Exists(paths.ImagePreds), "image_preds.json");
        Assert.True(File.Exists(paths.Log), "job.log");
        Assert.Equal(WorkspacePaths.StatsFileName, manifest.Artifacts.Stats);
        Assert.Null(manifest.Artifacts.Render);   // not requested

        // a frame sequence is used in place: no second copy of the images on disk
        Assert.False(Directory.Exists(paths.FramesDir));

        // the job is discoverable by a plain directory scan, with no other state involved
        Assert.Equal(result.JobId, store.DiscoverAll().Single().Manifest.JobId);
    }

    [Fact]
    public async Task RunAsync_WritesResultsThatReopenWithTheRadarAndPitchOverlayIntact()
    {
        RequireAssets();

        var store = new WorkspaceStore(_root);
        var result = await Run(store, new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Path.Combine(_repo!, FramesDir),
            ConfigPath = Path.Combine(_repo!, ConfigPath),
            ModelsDir = Path.Combine(_repo!, ModelsDir),
            MaxFrames = 6,
        }, new SynchronousProgress(_ => { }), CancellationToken.None);

        var paths = store.PathsFor(result.JobDirectory);
        var (detections, imagePreds) = store.ReadFrames(paths);

        Assert.Equal(6, imagePreds.Count);
        Assert.NotEmpty(detections.Rows);

        // the two columns TrackerState.Save silently drops — without them a reopened job has
        // no radar and no pitch overlay, which is half the point of the tool
        Assert.Contains(detections.Rows, d => d["bbox_pitch"] is BboxPitch);
        Assert.Contains(imagePreds.Rows, i => i["lines"] is IReadOnlyDictionary<string, List<Keypoint>>);

        // and the frames are still addressable, which is what playback needs
        Assert.All(imagePreds.Rows, i => Assert.True(File.Exists(i.Get<string>("file_path"))));

        var stats = store.ReadStats(paths);
        Assert.Equal(6, stats.FramesProcessed);
    }

    [Fact]
    public async Task RunAsync_ReportsProgressThatEndsWhereItShould()
    {
        RequireAssets();

        var updates = new List<AnalysisProgress>();
        var store = new WorkspaceStore(_root);
        await Run(store, new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Path.Combine(_repo!, FramesDir),
            ConfigPath = Path.Combine(_repo!, ConfigPath),
            ModelsDir = Path.Combine(_repo!, ModelsDir),
            MaxFrames = 6,
        }, new SynchronousProgress(updates.Add), CancellationToken.None);

        Assert.Equal(AnalysisStage.Extracting, updates[0].Stage);
        Assert.Equal(AnalysisStage.Done, updates[^1].Stage);
        Assert.Equal(1.0, updates[^1].Fraction);

        // never goes backwards, whatever the weights think
        var fractions = updates.Select(u => u.Fraction).ToList();
        Assert.Equal(fractions.OrderBy(f => f), fractions);
    }

    [Fact]
    public async Task RunAsync_CancellationLeavesACancelledJobRatherThanAHalfWrittenOne()
    {
        RequireAssets();

        var store = new WorkspaceStore(_root);
        using var cts = new CancellationTokenSource();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            Run(store, new AnalysisRequest
            {
                SourceKind = SourceKind.FrameSequence,
                SourcePath = Path.Combine(_repo!, FramesDir),
                ConfigPath = Path.Combine(_repo!, ConfigPath),
                ModelsDir = Path.Combine(_repo!, ModelsDir),
                // enough frames that cancelling lands mid-module rather than near the end
                MaxFrames = 12,
            }, new SynchronousProgress(p =>
            {
                if (p.Stage == AnalysisStage.Tracking) cts.Cancel();
            }), cts.Token));

        var job = store.DiscoverAll().Single();
        Assert.Equal(JobStatus.Cancelled, job.Manifest.Status);
        Assert.NotNull(job.Manifest.CompletedUtc);
        Assert.False(File.Exists(job.Paths.Stats), "a cancelled run must not leave stats behind");
    }

    [Fact]
    public async Task RunAsync_FindsTheModelsFromAConfigWhateverTheWorkingDirectoryIs()
    {
        RequireAssets();

        // The app's working directory is its own bin folder, not the repo root the CLI is run
        // from, so `models_dir: pretrained_models` from the config would otherwise resolve
        // against bin\Debug and the run would die with a missing-model error.
        var store = new WorkspaceStore(_root);   // creates the root this test then moves into
        var previous = Directory.GetCurrentDirectory();
        Directory.SetCurrentDirectory(_root);
        try
        {
            var result = await Run(store, new AnalysisRequest
            {
                SourceKind = SourceKind.FrameSequence,
                SourcePath = Path.Combine(_repo!, FramesDir),
                ConfigPath = Path.Combine(_repo!, ConfigPath),
                // deliberately not set: the config's own relative models_dir must be enough
                MaxFrames = 4,
            }, new SynchronousProgress(_ => { }), CancellationToken.None);

            var manifest = store.ReadManifest(result.JobDirectory);
            Assert.Equal(JobStatus.Completed, manifest.Status);
            Assert.Contains("YoloOnnxDetector", manifest.PipelineDescription);
        }
        finally
        {
            Directory.SetCurrentDirectory(previous);
        }
    }

    [Fact]
    public async Task RunAsync_RecordsWhyAJobFailedInsteadOfLosingIt()
    {
        var store = new WorkspaceStore(_root);

        await Assert.ThrowsAnyAsync<Exception>(() => Run(store,
            new AnalysisRequest
            {
                SourceKind = SourceKind.FrameSequence,
                SourcePath = Path.Combine(_root, "does-not-exist"),
            }, new SynchronousProgress(_ => { }), CancellationToken.None));

        var job = store.DiscoverAll().Single();
        Assert.Equal(JobStatus.Failed, job.Manifest.Status);
        Assert.NotNull(job.Manifest.FailureMessage);
        Assert.Contains("does-not-exist", job.Manifest.FailureMessage);
    }

    private sealed class SynchronousProgress(Action<AnalysisProgress> handler) : IProgress<AnalysisProgress>
    {
        public void Report(AnalysisProgress value) => handler(value);
    }
}
