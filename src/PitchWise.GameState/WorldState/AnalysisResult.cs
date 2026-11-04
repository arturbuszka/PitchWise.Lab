using System.Text.Json;
using System.Text.Json.Serialization;
using PitchWise.Engine;

namespace PitchWise.GameState.WorldState;

/// <summary>An engine event in the form other programs read it.</summary>
public sealed record AnalysisEvent(
    string Type,
    double TimestampSeconds,
    double Confidence,
    string? Label,
    int? PlayerId,
    string Team);

/// <summary>
/// What the <c>analyze</c> command writes to <c>result.json</c>. This is the contract with programs
/// that run the CLI as a separate process, so a breaking change must raise <see cref="CurrentSchemaVersion"/>.
/// </summary>
public sealed record AnalysisResult(
    int SchemaVersion,
    double Fps,
    MatchStatsReport Stats,
    IReadOnlyList<AnalysisEvent> Events)
{
    public const int CurrentSchemaVersion = 1;

    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) },
    };

    /// <summary>Builds the result, keeping only events a consumer can store. Passes and possession
    /// changes are internal signals, already counted in <paramref name="stats"/>.</summary>
    public static AnalysisResult Create(double fps, MatchStatsReport stats, IEnumerable<GameEvent> events) =>
        new(CurrentSchemaVersion, fps, stats, events
            .Where(e => GameEventType.PersistableTypes.Contains(e.Type))
            .OrderBy(e => e.TimestampSeconds)
            .Select(e => new AnalysisEvent(e.Type, e.TimestampSeconds, e.Confidence, e.Label, e.PlayerId,
                e.Team.ToString().ToLowerInvariant()))
            .ToList());
}
