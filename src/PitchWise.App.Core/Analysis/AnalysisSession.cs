using PitchWise.App.Core.Playback;
using PitchWise.App.Core.Workspace;
using PitchWise.GameState.Dataset;
using PitchWise.GameState.WorldState;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Analysis;

/// <summary>
/// One finished analysis, loaded and ready to browse: every panel reads from this.
/// </summary>
/// <remarks>
/// Loading is deliberately eager and done once, off the UI thread — reading a few hundred
/// milliseconds of json beats making every panel handle a half-loaded session.
/// </remarks>
public sealed class AnalysisSession : IDisposable
{
    private AnalysisSession(AnalysisManifest manifest, WorkspacePaths paths, DataFrame detections,
        DataFrame imagePreds, DataFrame? groundTruth, MatchStatsReport stats, IFrameSource frames)
    {
        Manifest = manifest;
        Paths = paths;
        Detections = detections;
        ImagePreds = imagePreds;
        GroundTruth = groundTruth;
        Stats = stats;
        Frames = frames;
        Overlay = new OverlayRenderer(detections, groundTruth);
    }

    public AnalysisManifest Manifest { get; }
    public WorkspacePaths Paths { get; }
    public DataFrame Detections { get; }
    public DataFrame ImagePreds { get; }

    /// <summary>Dataset ground truth, for SoccerNet jobs only.</summary>
    public DataFrame? GroundTruth { get; }

    public MatchStatsReport Stats { get; }
    public IFrameSource Frames { get; }
    public OverlayRenderer Overlay { get; }

    /// <summary>The recorded observation stream — the input to every re-tune.</summary>
    public string WorldStateJsonlPath => Paths.WorldStateJsonl;

    /// <summary>Loads a completed job. Call off the UI thread.</summary>
    /// <remarks>
    /// The manifest is re-read here rather than taken from the caller: the runner writes the job's
    /// final state to disk, so the file is the truth about what a job actually produced, and an
    /// in-memory copy taken before the run finished would disagree with the artifacts sitting
    /// beside it.
    /// </remarks>
    public static AnalysisSession Open(IWorkspaceStore store, WorkspacePaths paths,
        IFrameSourceFactory frameSources)
    {
        var manifest = store.ReadManifest(paths.JobDirectory);
        if (manifest.Status != JobStatus.Completed)
            throw new InvalidOperationException(
                $"'{manifest.DisplayName}' is {manifest.Status}; only a completed job has results");

        var (detections, imagePreds) = store.ReadFrames(paths);
        var stats = store.ReadStats(paths);
        var groundTruth = TryLoadGroundTruth(manifest);
        var frames = frameSources.Create(manifest, imagePreds);

        return new AnalysisSession(manifest, paths, detections, imagePreds, groundTruth, stats, frames);
    }

    /// <summary>
    /// Re-reads the dataset's ground truth for SoccerNet jobs, which is the only way to get the
    /// comparison radar. Absence is normal — most jobs have none — so failure is not fatal.
    /// </summary>
    private static DataFrame? TryLoadGroundTruth(AnalysisManifest manifest)
    {
        if (manifest.SourceKind != SourceKind.SoccerNetVideo || !Directory.Exists(manifest.SourcePath))
            return null;
        try
        {
            var set = SoccerNetGameState.LoadVideo(manifest.SourcePath);
            return set.DetectionsGt.IsEmpty ? null : set.DetectionsGt;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or KeyNotFoundException)
        {
            return null;
        }
    }

    public void Dispose()
    {
        Overlay.Dispose();
        Frames.Dispose();
    }
}
