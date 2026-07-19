using PitchWise.App.Core.Analysis;
using PitchWise.GameState.WorldState;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Workspace;

/// <summary>
/// Owns the on-disk workspace: creating job folders, reading and writing manifests and
/// artifacts, and discovering past jobs.
/// </summary>
public interface IWorkspaceStore
{
    /// <summary>Root under which every job folder lives.</summary>
    string Root { get; }

    /// <summary>Creates a job folder and writes its Queued manifest, so an interrupted run
    /// leaves a tombstone on disk rather than vanishing.</summary>
    (WorkspacePaths Paths, AnalysisManifest Manifest) CreateJob(AnalysisRequest request);

    /// <summary>Rewrites a manifest atomically.</summary>
    void SaveManifest(WorkspacePaths paths, AnalysisManifest manifest);

    AnalysisManifest ReadManifest(string jobDirectory);

    /// <summary>Every job in the workspace, newest first. Unreadable folders are skipped
    /// rather than failing the scan.</summary>
    IReadOnlyList<(WorkspacePaths Paths, AnalysisManifest Manifest)> DiscoverAll();

    WorkspacePaths PathsFor(string jobDirectory);

    void WriteStats(WorkspacePaths paths, MatchStatsReport report);
    MatchStatsReport ReadStats(WorkspacePaths paths);

    void WriteFrames(WorkspacePaths paths, DataFrame detections, DataFrame imagePreds);
    (DataFrame Detections, DataFrame ImagePreds) ReadFrames(WorkspacePaths paths);

    /// <summary>Deletes a job folder and everything in it.</summary>
    void DeleteJob(WorkspacePaths paths);

    /// <summary>Drops the extracted frames of a job, keeping its analysis artifacts.
    /// Playback then falls back to the source video, whose seeking is approximate.</summary>
    void DeleteFrames(WorkspacePaths paths);
}
