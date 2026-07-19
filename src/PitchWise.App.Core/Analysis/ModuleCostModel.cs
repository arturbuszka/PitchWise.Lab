using PitchWise.Tracking.Engine;

namespace PitchWise.App.Core.Analysis;

/// <summary>
/// Turns the engine's module-major ticks into a fraction a progress bar can show.
/// </summary>
/// <remarks>
/// The engine reports "module 3 of 8, frame 200 of 750", which is honest but useless raw:
/// modules differ in cost per frame by an order of magnitude, so weighting each equally would
/// make the bar crawl through the detector and then leap through the tracklet modules.
///
/// The weights are a display heuristic — approximate, and wrong the moment someone swaps a model
/// — so the fraction is never allowed to move backwards: <see cref="Advance"/> clamps to the
/// highest value seen. A bar that lies slightly about speed is fine; one that goes backwards
/// reads as a bug.
/// </remarks>
public sealed class ModuleCostModel
{
    /// <summary>
    /// Relative cost per frame, keyed by a substring of the module name so variants
    /// (YoloOnnxDetector, PrtReidOnnx, …) still match. Ordered by rough measured cost;
    /// anything unlisted is cheap.
    /// </summary>
    private static readonly (string Match, double Weight)[] Weights =
    [
        ("Yolo", 40),
        ("Reid", 25),
        ("Mmocr", 20),
        ("Keypoints", 10),
        ("Calib", 4),
        ("StrongSort", 3),
    ];

    private const double DefaultWeight = 1;

    private readonly double[] _weights;
    private readonly double _totalWeight;
    private double _highWaterMark;

    /// <param name="moduleNames">
    /// The pipeline's modules, in order. Known up front by whoever built the pipeline, which is
    /// what lets the already-finished modules be charged at their real weight instead of an
    /// average.
    /// </param>
    public ModuleCostModel(IReadOnlyList<string> moduleNames)
    {
        _weights = moduleNames.Select(WeightOf).ToArray();
        _totalWeight = _weights.Sum();
    }

    /// <summary>Cost weight of one module, by name.</summary>
    public static double WeightOf(string moduleName)
    {
        foreach (var (match, weight) in Weights)
            if (moduleName.Contains(match, StringComparison.OrdinalIgnoreCase))
                return weight;
        return DefaultWeight;
    }

    /// <summary>Fraction of the whole run a tick represents, ignoring monotonicity.</summary>
    public double RawFraction(TrackingProgress progress)
    {
        if (_totalWeight <= 0 || progress.ModuleIndex >= _weights.Length) return 0;

        var done = 0.0;
        for (var i = 0; i < progress.ModuleIndex; i++) done += _weights[i];
        done += _weights[progress.ModuleIndex] * progress.ModuleFraction;
        return Math.Clamp(done / _totalWeight, 0, 1);
    }

    /// <summary>Folds a tick into a monotonic overall fraction.</summary>
    public double Advance(TrackingProgress progress)
        => _highWaterMark = Math.Max(_highWaterMark, RawFraction(progress));

    /// <summary>Builds the update the UI binds to.</summary>
    public AnalysisProgress ToAnalysisProgress(TrackingProgress progress, TimeSpan elapsed)
    {
        var fraction = Advance(progress);
        return new AnalysisProgress(
            AnalysisStage.Tracking,
            $"{progress.ModuleName} ({progress.ModuleIndex + 1}/{progress.ModuleCount})",
            fraction,
            progress.ItemsDone,
            progress.ItemsTotal,
            elapsed,
            EstimateEta(fraction, elapsed));
    }

    /// <summary>Reports a stage outside the tracking loop, keeping the bar monotonic.</summary>
    public AnalysisProgress ToStageProgress(AnalysisStage stage, string detail, double fraction,
        TimeSpan elapsed)
    {
        _highWaterMark = Math.Max(_highWaterMark, fraction);
        return new AnalysisProgress(stage, detail, _highWaterMark, 0, 0, elapsed, null);
    }

    /// <summary>
    /// Elapsed time extrapolated to the whole run. Suppressed below 5%, where the estimate is
    /// dominated by model loading and would swing wildly.
    /// </summary>
    private static TimeSpan? EstimateEta(double fraction, TimeSpan elapsed)
        => fraction is > 0.05 and < 1
            ? TimeSpan.FromSeconds(elapsed.TotalSeconds * (1 - fraction) / fraction)
            : null;
}
