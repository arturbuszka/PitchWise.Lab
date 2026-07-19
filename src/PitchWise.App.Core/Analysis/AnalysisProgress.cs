namespace PitchWise.App.Core.Analysis;

/// <summary>Coarse phase of a job, for a label the user can read at a glance.</summary>
public enum AnalysisStage
{
    Queued,
    Extracting,
    Tracking,
    Observing,
    Rendering,
    Done,
}

/// <summary>One progress update for a running job, in terms a progress bar can use.</summary>
/// <param name="Stage">Coarse phase.</param>
/// <param name="Detail">Human-readable specifics, e.g. "YoloOnnxDetector (1/8)".</param>
/// <param name="Fraction">Overall completion 0..1. Weighted and monotonic by construction.</param>
/// <param name="ItemsDone">Items the current module has finished.</param>
/// <param name="ItemsTotal">Items the current module will process.</param>
/// <param name="Elapsed">Wall-clock time since the job started.</param>
/// <param name="Eta">Estimated time remaining, or null while the estimate is still noise.</param>
public readonly record struct AnalysisProgress(
    AnalysisStage Stage,
    string Detail,
    double Fraction,
    long ItemsDone,
    long ItemsTotal,
    TimeSpan Elapsed,
    TimeSpan? Eta);
