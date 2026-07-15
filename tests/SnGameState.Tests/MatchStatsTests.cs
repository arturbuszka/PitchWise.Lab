using PitchWise.Engine;
using PitchWise.Engine.Rules;
using SnGameState.WorldState;

namespace SnGameState.Tests;

/// <summary>
/// Covers the world-state statistics layer: the ported engine still folds observations (smoke),
/// and the new sn-gamestate distance/speed/possession aggregation is correct on synthetic input.
/// Pitch coordinates are in metres, centre-origin — as the vision adapter produces them.
/// </summary>
public class MatchStatsTests
{
    private const double Fps = 10.0;
    private const double Dt = 1.0 / Fps;

    /// <summary>Builds a frame with one ball and an arbitrary set of players.</summary>
    private static FrameObservation Frame(int i, (double x, double y)? ball, params PlayerObservation[] players) =>
        new(i, i * Dt, players,
            ball is { } b ? new BallObservation(b.x, b.y, 0.9, true) : BallObservation.Missing,
            NormalizedCoords: false);

    private static PlayerObservation P(int track, TeamId team, double x, double y) =>
        new(PlayerIdentity.FromTrack(track), PlayerRole.Player, team, x, y, 0.9);

    [Fact]
    public void EnginePort_FoldsObservations_WithoutThrowing()
    {
        // Smoke: the copied engine builds a coherent world state from a trivial stream.
        var builder = new WorldStateBuilder();
        PitchWise.Engine.WorldState? last = null;
        for (int i = 0; i < 5; i++)
            last = builder.Add(Frame(i, (0, 0), P(1, TeamId.A, 0.5, 0)));

        Assert.NotNull(last);
        Assert.Equal(4, last!.FrameIndex);
        Assert.True(last.Reliable);
    }

    [Fact]
    public void Possession_AttributedToControllingTeam()
    {
        // A single team-A player sits on the ball for the whole clip: possession should be ~100% A.
        var frames = new List<FrameObservation>();
        for (int i = 0; i < 40; i++)
            frames.Add(Frame(i, (0, 0),
                P(1, TeamId.A, 0.3, 0),        // on the ball
                P(2, TeamId.B, 20, 0)));       // far away

        var report = new MatchStatsCollector().Run(frames);

        Assert.True(report.ControlledSeconds > 0, "expected some controlled time");
        Assert.True(report.TeamA.PossessionPct > 90, $"A possession was {report.TeamA.PossessionPct}");
        Assert.Equal(0, report.TeamB.Passes);
    }

    [Fact]
    public void Pass_BetweenTeammates_IsCounted()
    {
        var frames = new List<FrameObservation>();
        int f = 0;

        // Phase 1: A-player #1 controls the ball at x=0 for long enough to capture possession.
        for (int i = 0; i < 15; i++, f++)
            frames.Add(Frame(f, (0, 0), P(1, TeamId.A, 0.3, 0), P(2, TeamId.A, 20, 0), P(3, TeamId.B, -20, 0)));

        // Phase 2: the ball travels to team-mate #2 at x=20 (well over PassMinDistance = 3 m).
        for (int i = 1; i <= 6; i++, f++)
        {
            double bx = i * (20.0 / 6);
            frames.Add(Frame(f, (bx, 0), P(1, TeamId.A, 0.3, 0), P(2, TeamId.A, 20, 0), P(3, TeamId.B, -20, 0)));
        }

        // Phase 3: #2 now controls the ball at x=20 long enough to be the new owner.
        for (int i = 0; i < 15; i++, f++)
            frames.Add(Frame(f, (20, 0), P(1, TeamId.A, 0.3, 0), P(2, TeamId.A, 20, 0), P(3, TeamId.B, -20, 0)));

        var report = new MatchStatsCollector().Run(frames);

        Assert.True(report.TeamA.Passes >= 1, $"expected a completed A pass, got {report.TeamA.Passes}");
        Assert.Equal(0, report.TeamA.Turnovers);
    }

    [Fact]
    public void Distance_IntegratesTrajectory_AndRejectsTeleports()
    {
        // Player #1 walks 1 m per frame (10 m/s at 10 fps → within cap) for 10 frames = 10 m,
        // then teleports 50 m in one frame (a re-track artefact) which must NOT be counted.
        var frames = new List<FrameObservation>();
        double x = 0;
        int i = 0;
        for (; i < 11; i++, x += 1.0)
            frames.Add(Frame(i, (100, 100), P(1, TeamId.A, x, 0)));   // ball parked far away
        // teleport
        frames.Add(Frame(i, (100, 100), P(1, TeamId.A, x + 50, 0)));

        var report = new MatchStatsCollector().Run(frames);
        var player = Assert.Single(report.Players);

        // 10 steps of 1 m = 10 m; the 50 m jump is discarded.
        Assert.Equal(10.0, player.DistanceMeters, precision: 1);
        // 1 m / 0.1 s = 10 m/s = 36 km/h; teleport (500 m/s) rejected, so top speed stays 36.
        Assert.Equal(36.0, player.TopSpeedKmh, precision: 1);
    }

    [Fact]
    public void Sprint_CountedOnlyWhenSustained()
    {
        // Player runs at 8 m/s (> 7 m/s sprint threshold) for 2 s (sustained → 1 sprint).
        var frames = new List<FrameObservation>();
        double x = 0;
        int i = 0;
        for (; i < 21; i++, x += 8.0 * Dt)   // 8 m/s
            frames.Add(Frame(i, (200, 200), P(1, TeamId.A, x, 0)));

        var report = new MatchStatsCollector().Run(frames);
        var player = Assert.Single(report.Players);

        Assert.Equal(1, player.Sprints);
        Assert.True(player.TopSpeedKmh >= 28, $"top speed {player.TopSpeedKmh} km/h");
        Assert.Equal(1, player.TrackId);
    }
}
