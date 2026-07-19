using PitchWise.App.Core.Workspace;

namespace PitchWise.App.Core.Analysis;

/// <summary>Where a finished job's artifacts ended up.</summary>
/// <param name="JobId">Identifies the job folder in the workspace.</param>
/// <param name="JobDirectory">Absolute path to that folder.</param>
public readonly record struct AnalysisResult(string JobId, string JobDirectory);

/// <summary>
/// Runs one analysis end to end: source to frames, frames through the pipeline, detections to
/// observations, observations to statistics and a replayable dump.
/// </summary>
public interface IAnalysisJobRunner
{
    /// <summary>
    /// Runs an already-created job on a background thread, reporting progress as it goes.
    /// </summary>
    /// <param name="paths">
    /// The job folder to work in, from <see cref="Workspace.IWorkspaceStore.CreateJob"/>. The
    /// runner does not create the job: the queue has to show it — and leave a crash tombstone for
    /// it — from the moment it is queued, which may be long before the runner gets to it.
    /// </param>
    /// <param name="progress">Reported from the worker thread.</param>
    /// <exception cref="OperationCanceledException">The token was signalled.</exception>
    Task<AnalysisResult> RunAsync(WorkspacePaths paths, AnalysisRequest request,
        IProgress<AnalysisProgress> progress, CancellationToken ct);
}
