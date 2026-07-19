using PitchWise.Engine;

namespace PitchWise.App.Core.Stats;

/// <summary>One clickable mark on the timeline.</summary>
/// <param name="Type">The engine's event type, e.g. "pass" or "possession_change".</param>
/// <param name="TimestampSeconds">When it happened, as the engine saw it.</param>
/// <param name="FrameIndex">Where to seek to show it.</param>
/// <param name="Team">Which side it belongs to; drives the marker's colour.</param>
/// <param name="PlayerId">Who, when the rule could attribute it.</param>
/// <param name="Label">Short text for the marker's tooltip.</param>
/// <param name="Confidence">The rule's own confidence, 0..1.</param>
public readonly record struct TimelineMarker(
    string Type,
    double TimestampSeconds,
    int FrameIndex,
    TeamId Team,
    int? PlayerId,
    string Label,
    double Confidence);

/// <summary>
/// Turns replayed engine events into timeline markers.
/// </summary>
/// <remarks>
/// The timeline is projected from a WorldStateReplay of the recorded observations — not from a
/// second pass over the video — which is what makes it re-derivable in milliseconds when a
/// threshold changes. Using the same rule set as MatchStatsCollector means the timeline and the
/// stats panel agree by construction rather than by being kept in step by hand.
/// </remarks>
public static class TimelineProjector
{
    /// <summary>The rules MatchStatsCollector runs. Both panels must use this same set.</summary>
    public static IReadOnlyList<IGameRule> Rules(EngineConfig cfg) =>
        [new PitchWise.Engine.Rules.PassRule(cfg), new PitchWise.Engine.Rules.PossessionRule()];

    /// <summary>Projects a replay's events onto the video's frames.</summary>
    /// <remarks>
    /// A GameEvent carries only a timestamp, so the dump header's fps is the only bridge back to
    /// a frame index. That mapping assumes the analysis kept every frame — see
    /// AnalysisRequest.FrameStride for why stride is pinned to 1.
    /// </remarks>
    public static IReadOnlyList<TimelineMarker> Project(WorldStateReplay.Result replay)
    {
        var fps = replay.Header.Fps > 0 ? replay.Header.Fps : 25;
        return replay.Events
            .Select(e => new TimelineMarker(
                e.Type,
                e.TimestampSeconds,
                (int)Math.Round(e.TimestampSeconds * fps),
                e.Team,
                e.PlayerId,
                Describe(e),
                e.Confidence))
            .OrderBy(m => m.TimestampSeconds)
            .ToList();
    }

    private static string Describe(GameEvent e)
    {
        var who = e.PlayerId is { } id ? $" #{id}" : "";
        var team = e.Team switch
        {
            TeamId.A => "A",
            TeamId.B => "B",
            _ => "?",
        };
        var what = e.Type switch
        {
            GameEventType.Pass => "Pass",
            GameEventType.WaywardPass => "Turnover",
            GameEventType.PossessionChange => "Possession",
            GameEventType.Goal => "Goal",
            GameEventType.Shot => "Shot",
            _ => e.Type,
        };
        return e.Label is { Length: > 0 } label
            ? $"{what} {team}{who} — {label}"
            : $"{what} {team}{who}";
    }
}
