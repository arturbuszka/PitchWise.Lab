using System.Diagnostics;
using PitchWise.App.Core.Stats;
using PitchWise.Engine;
using PitchWise.GameState.WorldState;

namespace PitchWise.App.Core.Tuning;

/// <param name="Markers">Timeline events under the config that was replayed.</param>
/// <param name="Stats">Match statistics under that same config.</param>
/// <param name="FramesReplayed">How many observation frames the dump held.</param>
/// <param name="Duration">How long the replay took — expected to be milliseconds.</param>
public sealed record TuningResult(
    IReadOnlyList<TimelineMarker> Markers,
    MatchStatsReport Stats,
    int FramesReplayed,
    TimeSpan Duration);

/// <summary>Re-derives events and statistics under a different engine configuration.</summary>
public interface ITuningService
{
    /// <summary>
    /// Replays the recorded observations through a fresh engine under <paramref name="cfg"/>.
    /// </summary>
    /// <remarks>
    /// No video decode and no ONNX — this is what WorldStateReplay exists for. The dump holds
    /// observations rather than conclusions, so the ball filter and possession state machine are
    /// rebuilt each time and their thresholds are genuinely tunable.
    /// </remarks>
    Task<TuningResult> ReplayAsync(string jsonlPath, EngineConfig cfg, CancellationToken ct);
}

/// <inheritdoc />
public sealed class TuningService : ITuningService
{
    public Task<TuningResult> ReplayAsync(string jsonlPath, EngineConfig cfg, CancellationToken ct)
        // off the UI thread for cancellation and responsiveness, not because it is slow: a line
        // per frame through a Kalman update and a few rule checks is milliseconds even for a
        // whole match, which is what lets a slider re-run the entire thing on every change
        => Task.Run(() => Replay(jsonlPath, cfg, ct), ct);

    private static TuningResult Replay(string jsonlPath, EngineConfig cfg, CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();

        var replay = WorldStateReplay.Run(jsonlPath, TimelineProjector.Rules(cfg), cfg);
        ct.ThrowIfCancellationRequested();

        // a second pass over the same dump: MatchStatsCollector owns its own engine and rules,
        // and re-deriving the report is the only way the stats panel reflects the new config
        var stats = new MatchStatsCollector(cfg).Run(WorldStateReplay.ReadObservations(jsonlPath));

        return new TuningResult(TimelineProjector.Project(replay), stats, replay.FramesReplayed,
            stopwatch.Elapsed);
    }
}
