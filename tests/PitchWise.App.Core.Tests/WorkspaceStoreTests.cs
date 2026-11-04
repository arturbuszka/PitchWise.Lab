using PitchWise.App.Core.Analysis;
using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.Tests;

public class WorkspaceStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"ws-test-{Guid.NewGuid():N}");
    private readonly WorkspaceStore _store;

    public WorkspaceStoreTests() => _store = new WorkspaceStore(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    /// <summary>Mirrors the real layout: frames sit in &lt;video&gt;/img1.</summary>
    private static AnalysisRequest Request(string name = "SNGS-021") => new()
    {
        SourceKind = SourceKind.FrameSequence,
        SourcePath = Path.Combine(Path.GetTempPath(), "pw-tests", name, "img1"),
        MaxFrames = 20,
    };

    [Fact]
    public void CreateJob_LeavesAQueuedTombstoneOnDiskBeforeAnyWorkHappens()
    {
        var (paths, manifest) = _store.CreateJob(Request());

        // the manifest exists from the moment the job is queued, so a crash mid-run leaves
        // a job that can be found and reported rather than an unexplained folder
        Assert.True(File.Exists(paths.Manifest));
        Assert.Equal(JobStatus.Queued, manifest.Status);
        Assert.Equal(JobStatus.Queued, _store.ReadManifest(paths.JobDirectory).Status);
        Assert.Equal(AnalysisManifest.CurrentSchemaVersion, manifest.SchemaVersion);
    }

    [Fact]
    public void Manifest_RoundTripsThroughDiskIncludingEnumsAndArtifacts()
    {
        var (paths, manifest) = _store.CreateJob(Request());

        var updated = manifest with
        {
            Status = JobStatus.Completed,
            CompletedUtc = DateTimeOffset.UtcNow,
            FrameCount = 750,
            Fps = 25,
            ImWidth = 1920,
            ImHeight = 1080,
            PipelineDescription = "YoloOnnxDetector -> StrongSortModule",
            Artifacts = new ArtifactSet { Detections = "detections.json", Stats = "stats.json" },
        };
        _store.SaveManifest(paths, updated);
        var back = _store.ReadManifest(paths.JobDirectory);

        Assert.Equal(updated, back);
        Assert.Equal(SourceKind.FrameSequence, back.SourceKind);
        Assert.Equal("detections.json", back.Artifacts.Detections);
        Assert.Null(back.Artifacts.Render);

        // enums are written by name: a manifest is meant to be read by a human when a run goes wrong
        Assert.Contains("\"Completed\"", File.ReadAllText(paths.Manifest));
    }

    [Fact]
    public void SaveManifest_DoesNotLeaveATempFileBehind()
    {
        var (paths, manifest) = _store.CreateJob(Request());
        _store.SaveManifest(paths, manifest with { Status = JobStatus.Running });

        Assert.Empty(Directory.GetFiles(paths.JobDirectory, "*.tmp"));
    }

    [Fact]
    public void DiscoverAll_ReturnsEveryJobNewestFirst()
    {
        var first = _store.CreateJob(Request("alpha"));
        _store.SaveManifest(first.Paths,
            first.Manifest with { CreatedUtc = DateTimeOffset.UtcNow.AddHours(-2) });
        var second = _store.CreateJob(Request("beta"));

        var found = _store.DiscoverAll();

        Assert.Equal(2, found.Count);
        Assert.Equal(second.Manifest.JobId, found[0].Manifest.JobId);
        Assert.Equal("alpha", found[1].Manifest.DisplayName);
    }

    [Fact]
    public void DiscoverAll_SkipsRubbishRatherThanHidingEveryOtherJob()
    {
        var good = _store.CreateJob(Request("good"));

        // a corrupt manifest, and a stray folder that is not a job at all
        var corrupt = Path.Combine(_root, "corrupt-job");
        Directory.CreateDirectory(corrupt);
        File.WriteAllText(Path.Combine(corrupt, WorkspacePaths.ManifestFileName), "{ this is not json");
        Directory.CreateDirectory(Path.Combine(_root, "not-a-job"));

        var found = _store.DiscoverAll();

        Assert.Single(found);
        Assert.Equal(good.Manifest.JobId, found[0].Manifest.JobId);
    }

    [Fact]
    public void ReadManifest_RefusesASchemaFromTheFuture()
    {
        var (paths, manifest) = _store.CreateJob(Request());
        _store.SaveManifest(paths, manifest with { SchemaVersion = AnalysisManifest.CurrentSchemaVersion + 1 });

        var ex = Assert.Throws<InvalidDataException>(() => _store.ReadManifest(paths.JobDirectory));
        Assert.Contains("schema", ex.Message);
    }

    [Fact]
    public void JobIds_AreUniqueEvenForTheSameSourceInTheSameSecond()
    {
        var a = _store.CreateJob(Request());
        var b = _store.CreateJob(Request());

        Assert.NotEqual(a.Manifest.JobId, b.Manifest.JobId);
        Assert.Equal(2, _store.DiscoverAll().Count);
        // still readable: the display name leads the folder name
        Assert.StartsWith("SNGS-021-", a.Manifest.JobId);
    }

    [Fact]
    public void DeleteFrames_KeepsTheAnalysisButDropsTheImages()
    {
        var (paths, _) = _store.CreateJob(Request());
        Directory.CreateDirectory(paths.FramesDir);
        File.WriteAllText(Path.Combine(paths.FramesDir, "000001.jpg"), "pretend frame");
        File.WriteAllText(paths.Stats, "{}");

        _store.DeleteFrames(paths);

        Assert.False(Directory.Exists(paths.FramesDir));
        Assert.True(File.Exists(paths.Stats));
        Assert.True(File.Exists(paths.Manifest));
    }

    [Fact]
    public void DeleteJob_RemovesTheFolderAndTheJobStopsBeingDiscovered()
    {
        var (paths, _) = _store.CreateJob(Request());

        _store.DeleteJob(paths);

        Assert.False(Directory.Exists(paths.JobDirectory));
        Assert.Empty(_store.DiscoverAll());
        _store.DeleteJob(paths);   // idempotent: deleting twice is not an error
    }

    /// <summary>An absolute path on whichever OS runs the tests; "d:\\x" is relative on Linux.</summary>
    private static string Rooted(params string[] parts) =>
        Path.Combine([Path.GetPathRoot(Path.GetTempPath())!, .. parts]);

    [Fact]
    public void ResolveDisplayName_NamesJobsAfterTheirSource()
    {
        Assert.Equal("SNGS-021", new AnalysisRequest
        {
            SourceKind = SourceKind.SoccerNetVideo,
            SourcePath = Rooted("data", "SoccerNetGS", "valid", "SNGS-021"),
        }.ResolveDisplayName());

        Assert.Equal("match-clip", new AnalysisRequest
        {
            SourceKind = SourceKind.VideoFile,
            SourcePath = Rooted("videos", "match-clip.mp4"),
        }.ResolveDisplayName());

        // SoccerNet frames live in <video>/img1, so the video is the useful name — and a
        // trailing separator must not swallow it
        Assert.Equal("SNGS-021", new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Rooted("data", "SNGS-021", "img1") + Path.DirectorySeparatorChar,
        }.ResolveDisplayName());

        // a frames folder that names itself keeps its own name
        Assert.Equal("second-half", new AnalysisRequest
        {
            SourceKind = SourceKind.FrameSequence,
            SourcePath = Rooted("captures", "second-half"),
        }.ResolveDisplayName());

        Assert.Equal("custom", new AnalysisRequest
        {
            SourceKind = SourceKind.VideoFile,
            SourcePath = Rooted("videos", "match.mp4"),
            DisplayName = "custom",
        }.ResolveDisplayName());
    }
}
