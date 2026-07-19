using System.Text.Json;
using System.Text.Json.Serialization;
using PitchWise.App.Core.Analysis;
using PitchWise.GameState.WorldState;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Workspace;

/// <summary>
/// The workspace on disk: one self-describing folder per job under <see cref="Root"/>.
/// </summary>
public sealed class WorkspaceStore : IWorkspaceStore
{
    private static readonly JsonSerializerOptions ManifestOptions = new()
    {
        WriteIndented = true,   // these get read by hand when something goes wrong
        Converters = { new JsonStringEnumConverter() },
    };

    public WorkspaceStore(string? root = null)
    {
        Root = root ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "PitchWise", "workspaces");
        Directory.CreateDirectory(Root);
    }

    public string Root { get; }

    public (WorkspacePaths Paths, AnalysisManifest Manifest) CreateJob(AnalysisRequest request)
    {
        var displayName = request.ResolveDisplayName();
        // readable prefix for humans browsing the folder, timestamp to sort by, short random
        // suffix so re-analysing the same source in the same second cannot collide
        var jobId = $"{Sanitize(displayName)}-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..6]}";

        var paths = new WorkspacePaths(Path.Combine(Root, jobId), jobId);
        Directory.CreateDirectory(paths.JobDirectory);

        var manifest = new AnalysisManifest
        {
            JobId = jobId,
            DisplayName = displayName,
            SourceKind = request.SourceKind,
            SourcePath = request.SourcePath,   // AnalysisRequest has already made these absolute
            CreatedUtc = DateTimeOffset.UtcNow,
            Status = JobStatus.Queued,
            FrameStride = request.FrameStride,
            MaxFrames = request.MaxFrames,
            StartSeconds = request.StartSeconds,
            Fps = request.SourceFps,
            ConfigPath = request.ConfigPath,
            ModelsDir = request.ModelsDir,
        };
        SaveManifest(paths, manifest);
        return (paths, manifest);
    }

    public void SaveManifest(WorkspacePaths paths, AnalysisManifest manifest)
    {
        // temp + move, so a crash mid-write cannot leave a half-written manifest that would
        // make the job unreadable on the next scan
        var temp = paths.Manifest + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(manifest, ManifestOptions));
        File.Move(temp, paths.Manifest, overwrite: true);
    }

    public AnalysisManifest ReadManifest(string jobDirectory)
    {
        var path = Path.Combine(jobDirectory, WorkspacePaths.ManifestFileName);
        var manifest = JsonSerializer.Deserialize<AnalysisManifest>(File.ReadAllText(path), ManifestOptions)
            ?? throw new InvalidDataException($"empty manifest: {path}");
        if (manifest.SchemaVersion > AnalysisManifest.CurrentSchemaVersion)
            throw new InvalidDataException(
                $"{path} is schema v{manifest.SchemaVersion}; this build reads up to " +
                $"v{AnalysisManifest.CurrentSchemaVersion}");
        return manifest;
    }

    public IReadOnlyList<(WorkspacePaths Paths, AnalysisManifest Manifest)> DiscoverAll()
    {
        var jobs = new List<(WorkspacePaths, AnalysisManifest)>();
        foreach (var directory in Directory.EnumerateDirectories(Root))
        {
            if (!File.Exists(Path.Combine(directory, WorkspacePaths.ManifestFileName))) continue;
            try
            {
                jobs.Add((PathsFor(directory), ReadManifest(directory)));
            }
            catch (Exception e) when (e is IOException or JsonException or InvalidDataException)
            {
                // one corrupt folder must not hide every other job
            }
        }
        return jobs.OrderByDescending(j => j.Item2.CreatedUtc).ToList();
    }

    public WorkspacePaths PathsFor(string jobDirectory)
        => new(jobDirectory, Path.GetFileName(Path.TrimEndingDirectorySeparator(jobDirectory)));

    public void WriteStats(WorkspacePaths paths, MatchStatsReport report)
        => File.WriteAllText(paths.Stats,
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

    public MatchStatsReport ReadStats(WorkspacePaths paths)
        => JsonSerializer.Deserialize<MatchStatsReport>(File.ReadAllText(paths.Stats))
           ?? throw new InvalidDataException($"empty stats: {paths.Stats}");

    public void WriteFrames(WorkspacePaths paths, DataFrame detections, DataFrame imagePreds)
    {
        DetectionsStore.WriteDetections(paths.Detections, detections);
        DetectionsStore.WriteImagePreds(paths.ImagePreds, imagePreds);
    }

    public (DataFrame Detections, DataFrame ImagePreds) ReadFrames(WorkspacePaths paths)
        => (DetectionsStore.ReadDetections(paths.Detections),
            DetectionsStore.ReadImagePreds(paths.ImagePreds));

    public void DeleteJob(WorkspacePaths paths)
    {
        if (Directory.Exists(paths.JobDirectory))
            Directory.Delete(paths.JobDirectory, recursive: true);
    }

    public void DeleteFrames(WorkspacePaths paths)
    {
        if (Directory.Exists(paths.FramesDir))
            Directory.Delete(paths.FramesDir, recursive: true);
    }

    private static string Sanitize(string name)
    {
        var clean = new string(name.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '-' : c).ToArray());
        return clean.Length > 40 ? clean[..40] : clean;
    }
}
