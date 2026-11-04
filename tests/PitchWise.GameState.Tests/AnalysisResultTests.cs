using System.Text.Json;
using PitchWise.Engine;
using PitchWise.GameState.WorldState;

namespace PitchWise.GameState.Tests;

public class AnalysisResultTests
{
    private static readonly MatchStatsReport Stats = new(10, 1, 0.5, 0.5,
        new TeamStats(60, 2, 1, 66.7), new TeamStats(40, 1, 0, 100), []);

    [Fact]
    public void Create_KeepsOnlyEventsAConsumerCanStore_InTimeOrder()
    {
        var result = AnalysisResult.Create(25, Stats,
        [
            new GameEvent(GameEventType.WaywardPass, 9, 0.8, "late", 7, TeamId.A),
            new GameEvent(GameEventType.Pass, 2, 0.9),
            new GameEvent(GameEventType.PossessionChange, 3, 0.9),
            new GameEvent(GameEventType.WaywardPass, 4, 0.7, "early"),
        ]);

        Assert.Equal(["early", "late"], result.Events.Select(e => e.Label));
    }

    [Fact]
    public void Json_UsesTheDocumentedFieldNames()
    {
        var result = AnalysisResult.Create(25, Stats,
            [new GameEvent(GameEventType.WaywardPass, 4, 0.7, "x", 7, TeamId.A)]);

        var root = JsonDocument.Parse(JsonSerializer.Serialize(result, AnalysisResult.JsonOptions)).RootElement;

        Assert.Equal(AnalysisResult.CurrentSchemaVersion, root.GetProperty("schema_version").GetInt32());
        Assert.Equal(60, root.GetProperty("stats").GetProperty("team_a").GetProperty("possession_pct").GetDouble());
        var ev = root.GetProperty("events")[0];
        Assert.Equal("wayward_pass", ev.GetProperty("type").GetString());
        Assert.Equal(4, ev.GetProperty("timestamp_seconds").GetDouble());
        Assert.Equal("a", ev.GetProperty("team").GetString());
    }
}
