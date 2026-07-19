namespace PitchWise.Tracking.Engine;

/// <summary>
/// One tick of a track run.
/// </summary>
/// <remarks>
/// The engine is module-major — the whole video passes through the detector before re-id starts —
/// so there is no single monotonic "frame N of M" to report. What a caller can be told is which
/// module is running, how far it has got through the video, and how far the run has got through
/// the module list. Overall progress is therefore ModuleIndex plus the within-module fraction:
/// honest, but NOT linear in time, because modules differ in cost per frame by an order of
/// magnitude. Weighting that into something a progress bar can show is the caller's job — the
/// engine reports what it knows, not what looks good.
/// </remarks>
/// <param name="VideoId">The video being tracked.</param>
/// <param name="ModuleName">Module currently running, e.g. "YoloOnnxDetector".</param>
/// <param name="ModuleIndex">Zero-based position of that module in the pipeline.</param>
/// <param name="ModuleCount">Total modules in the pipeline.</param>
/// <param name="ItemsDone">Items this module has finished (frames, or detections for
/// detection-level modules; 1 for the atomic video-level ones).</param>
/// <param name="ItemsTotal">Items this module will process in total.</param>
public readonly record struct TrackingProgress(
    long VideoId,
    string ModuleName,
    int ModuleIndex,
    int ModuleCount,
    long ItemsDone,
    long ItemsTotal)
{
    /// <summary>How far this module has got, 0..1. Zero when the total is unknown.</summary>
    public double ModuleFraction => ItemsTotal > 0 ? (double)ItemsDone / ItemsTotal : 0;
}
