using PitchWise.Engine;

namespace PitchWise.GameState.WorldState;

/// <summary>Per-team match aggregates. Possession and passing come from the engine's possession
/// state machine and PassRule; the split is over CONTROLLED time only, so A + B = 100.</summary>
public sealed record TeamStats(
    double PossessionPct,
    int Passes,
    int Turnovers,
    double PassAccuracyPct);

/// <summary>Per-player physical aggregates, integrated from the pitch trajectory (metres).</summary>
public sealed record PlayerStats(
    int PlayerId,
    int? TrackId,
    string Team,
    double DistanceMeters,
    double TopSpeedKmh,
    double AvgSpeedKmh,
    int Sprints,
    double TimeOnPitchSeconds);

/// <summary>The whole match report — the JSON the <c>stats</c> command writes.</summary>
public sealed record MatchStatsReport(
    int FramesProcessed,
    double DurationSeconds,
    double ControlledSeconds,
    double LooseSeconds,
    TeamStats TeamA,
    TeamStats TeamB,
    IReadOnlyList<PlayerStats> Players);
