using PitchWise.App.Core.Stats;
using PitchWise.App.Core.Tuning;
using PitchWise.App.Core.ViewModels;
using PitchWise.Engine;

namespace PitchWise.App.Core.Tests;

public class TimelineTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"timeline-test-{Guid.NewGuid():N}");

    public TimelineTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    /// <summary>
    /// Writes a dump of one team passing the ball between two players, which is the simplest
    /// stream that makes PassRule fire.
    /// </summary>
    private string WriteDump(double fps = 25)
    {
        var path = Path.Combine(_root, "stats.world_state.jsonl");
        using var dump = WorldStateJsonl.TryCreate(path,
            new WorldStateJsonl.Header(fps, FrameStride: 1, PitchLength: 105, PitchWidth: 68,
                NormalizedCoords: false))
            ?? throw new InvalidOperationException("could not create the dump");

        // player 1 holds the ball at x=-20, then it travels to player 2 at x=0
        for (var frame = 0; frame < 100; frame++)
        {
            var t = frame / fps;
            var ballX = frame < 30 ? -20 : frame < 50 ? -20 + (frame - 30) : 0;

            dump.Append(new FrameObservation(frame, t,
            [
                new PlayerObservation(1, PlayerRole.Player, TeamId.A, -20, 0, 0.9),
                new PlayerObservation(2, PlayerRole.Player, TeamId.A, 0, 0, 0.9),
                new PlayerObservation(3, PlayerRole.Player, TeamId.B, 30, 10, 0.9),
            ],
            new BallObservation(ballX, 0, 0.9, Detected: true),
            NormalizedCoords: false));
        }
        return path;
    }

    [Fact]
    public async Task Replay_DerivesTimelineAndStatsFromTheDumpInMilliseconds()
    {
        var path = WriteDump();

        var result = await new TuningService().ReplayAsync(path, new EngineConfig(), CancellationToken.None);

        Assert.Equal(100, result.FramesReplayed);
        // the whole point of replaying observations rather than re-running vision
        Assert.True(result.Duration < TimeSpan.FromSeconds(1),
            $"replaying 100 frames took {result.Duration.TotalMilliseconds:F0} ms");
        Assert.Equal(100, result.Stats.FramesProcessed);
        Assert.NotEmpty(result.Markers);
    }

    [Fact]
    public async Task Markers_MapBackToTheFrameTheEventHappenedOn()
    {
        const double fps = 25;
        var result = await new TuningService().ReplayAsync(WriteDump(fps), new EngineConfig(),
            CancellationToken.None);

        // a GameEvent carries only a timestamp; the dump header's fps is the only bridge back
        Assert.All(result.Markers, m =>
            Assert.Equal((int)Math.Round(m.TimestampSeconds * fps), m.FrameIndex));
        Assert.All(result.Markers, m => Assert.InRange(m.FrameIndex, 0, 100));
        Assert.Equal(result.Markers.OrderBy(m => m.TimestampSeconds), result.Markers);
    }

    [Fact]
    public async Task Timeline_AndStats_AgreeBecauseTheyRunTheSameRules()
    {
        var result = await new TuningService().ReplayAsync(WriteDump(), new EngineConfig(),
            CancellationToken.None);

        // if these ever diverge, one of the two panels is running a different rule set
        var passMarkers = result.Markers.Count(m => m.Type == GameEventType.Pass);
        Assert.Equal(result.Stats.TeamA.Passes + result.Stats.TeamB.Passes, passMarkers);

        var turnovers = result.Markers.Count(m => m.Type == GameEventType.WaywardPass);
        Assert.Equal(result.Stats.TeamA.Turnovers + result.Stats.TeamB.Turnovers, turnovers);
    }

    [Fact]
    public async Task Config_ChangesWhatTheTimelineShows()
    {
        var path = WriteDump();
        var service = new TuningService();

        var lenient = await service.ReplayAsync(path, new EngineConfig(), CancellationToken.None);
        // a pass must now travel further than this dump's ball ever does
        var strict = await service.ReplayAsync(path, new EngineConfig { PassMinDistance = 500 },
            CancellationToken.None);

        Assert.NotEqual(lenient.Markers.Count(m => m.Type == GameEventType.Pass),
            strict.Markers.Count(m => m.Type == GameEventType.Pass));
        Assert.DoesNotContain(strict.Markers, m => m.Type == GameEventType.Pass);
    }

    [Fact]
    public void Projector_LabelsMarkersWithTheirTeamAndPlayer()
    {
        var replay = new WorldStateReplay.Result(
            new WorldStateJsonl.Header(25, 1, 105, 68, false),
            [
                new GameEvent(GameEventType.Pass, 4.0, 0.9, PlayerId: 7, Team: TeamId.A),
                new GameEvent(GameEventType.WaywardPass, 2.0, 0.8, Team: TeamId.B),
            ],
            FramesReplayed: 100);

        var markers = TimelineProjector.Project(replay);

        // sorted by time regardless of the order the rules emitted them
        Assert.Equal([2.0, 4.0], markers.Select(m => m.TimestampSeconds));
        Assert.Equal("Turnover B", markers[0].Label);
        Assert.Equal("Pass A #7", markers[1].Label);
        Assert.Equal(50, markers[0].FrameIndex);
        Assert.Equal(100, markers[1].FrameIndex);
    }

    [Fact]
    public void ViewModel_FiltersMarkersAndSeeksWhenOneIsPicked()
    {
        var seeked = new List<int>();
        var vm = new TimelineViewModel(seeked.Add);

        vm.Update(
        [
            new TimelineMarker(GameEventType.Pass, 1, 25, TeamId.A, 1, "Pass A #1", 0.9),
            new TimelineMarker(GameEventType.WaywardPass, 2, 50, TeamId.B, null, "Turnover B", 0.8),
            new TimelineMarker(GameEventType.PossessionChange, 3, 75, TeamId.A, null, "Possession A", 0.7),
        ], totalFrames: 101);

        // possession changes are the noisiest, so they start hidden
        Assert.Equal(2, vm.Markers.Count);
        Assert.Equal("2 of 3 events", vm.SummaryText);

        vm.ShowPossessionChanges = true;
        Assert.Equal(3, vm.Markers.Count);

        vm.ShowPasses = false;
        Assert.DoesNotContain(vm.Markers, m => m.Marker.Type == GameEventType.Pass);

        // laid out proportionally along the video
        var turnover = vm.Markers.First(m => m.Marker.Type == GameEventType.WaywardPass);
        Assert.Equal(0.5, turnover.Position, precision: 3);

        vm.GoToCommand.Execute(turnover);
        Assert.Equal([50], seeked);
        Assert.Same(turnover, vm.SelectedMarker);
    }

    [Fact]
    public void ViewModel_ColorsMarkersByTeamAndSetsTurnoversApart()
    {
        var vm = new TimelineViewModel(_ => { });
        vm.Update(
        [
            new TimelineMarker(GameEventType.Pass, 1, 10, TeamId.A, null, "", 0.9),
            new TimelineMarker(GameEventType.Pass, 2, 20, TeamId.B, null, "", 0.9),
            new TimelineMarker(GameEventType.WaywardPass, 3, 30, TeamId.A, null, "", 0.9),
        ], totalFrames: 100);

        // team colours match the ellipses the overlay draws
        Assert.Equal("#c0392b", vm.Markers[0].Color);
        Assert.Equal("#2980b9", vm.Markers[1].Color);
        // a turnover is worth spotting whichever side lost the ball
        Assert.Equal("#e67e22", vm.Markers[2].Color);
    }

    [Fact]
    public void ReadObservations_RefusesAFileThatIsNotADump()
    {
        var path = Path.Combine(_root, "not-a-dump.jsonl");
        File.WriteAllText(path, "{\"f\":0,\"t\":0,\"p\":[],\"ball\":null,\"n\":false}\n");

        var ex = Assert.Throws<InvalidDataException>(
            () => WorldStateReplay.ReadObservations(path).ToList());
        Assert.Contains("header", ex.Message);
    }
}
