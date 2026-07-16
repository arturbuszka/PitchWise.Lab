using PitchWise.Engine;
using PitchWise.Engine.Rules;

namespace PitchWise.GameState.WorldState;

/// <summary>
/// Folds a stream of <see cref="FrameObservation"/> through the football engine and accumulates a
/// <see cref="MatchStatsReport"/>. Two families of statistic:
///
/// <list type="bullet">
/// <item><b>Possession &amp; passing</b> — read off the engine: each frame's elapsed time is
///       attributed to whichever team the <see cref="PossessionTracker"/> says is in control, and
///       pass/turnover counts come from <see cref="PassRule"/>'s events. This is the engine's
///       reason to exist; we only tally what it decides.</item>
/// <item><b>Distance &amp; speed</b> — integrated here from the per-player pitch trajectory. The
///       engine has no notion of these, so this is genuinely new PitchWise logic.</item>
/// </list>
///
/// The distance integrator rejects physically impossible frame-to-frame jumps: a re-track or a
/// homography wobble can teleport a foot-point by tens of metres, and counting that as "distance
/// covered" would inflate every total. A jump implying more than <see cref="MaxPlayerSpeed"/> is
/// dropped from the distance sum (but the player is still counted present).
/// </summary>
public sealed class MatchStatsCollector
{
    /// <summary>Speed above which a frame-to-frame move is treated as a tracking artefact, not
    /// running. 12 m/s ≈ 43 km/h — above any human sprint, so real play is never discarded.</summary>
    public const double MaxPlayerSpeed = 12.0;   // m/s

    /// <summary>Instantaneous speed at or above which the player is considered sprinting.
    /// ~25 km/h, the usual sports-science sprint threshold.</summary>
    public const double SprintSpeed = 7.0;       // m/s

    /// <summary>A sprint must be sustained this long to count, so a single noisy fast frame does
    /// not register as a sprint.</summary>
    public const double SprintMinDuration = 1.0; // s

    private readonly EngineConfig _cfg;
    private readonly WorldStateBuilder _builder;
    private readonly RuleEngine _rules;

    private int _frames;
    private double _firstTs = double.NaN;
    private double _lastTs = double.NaN;
    private double _prevTs = double.NaN;

    private double _controlledA, _controlledB, _loose;
    private int _passesA, _passesB, _turnoversA, _turnoversB;

    private readonly Dictionary<int, PlayerAccum> _players = new();

    public MatchStatsCollector(EngineConfig? cfg = null)
    {
        _cfg = cfg ?? new EngineConfig();
        _builder = new WorldStateBuilder(_cfg);
        _rules = new RuleEngine(new IGameRule[] { new PassRule(_cfg), new PossessionRule() }, _cfg);
    }

    /// <summary>Runs the whole stream and returns the report. The optional hook receives each
    /// derived <see cref="PitchWise.Engine.WorldState"/> (e.g. to also dump world_state.jsonl).</summary>
    public MatchStatsReport Run(IEnumerable<FrameObservation> observations,
        Action<FrameObservation, PitchWise.Engine.WorldState>? onFrame = null)
    {
        foreach (var obs in observations)
        {
            var state = _builder.Add(obs);
            foreach (var e in _rules.OnFrame(state)) Tally(e);
            Accumulate(obs, state);
            onFrame?.Invoke(obs, state);
        }
        foreach (var e in _rules.Flush()) Tally(e);
        return Build();
    }

    private void Accumulate(FrameObservation obs, PitchWise.Engine.WorldState state)
    {
        double ts = obs.TimestampSeconds;
        double dt = double.IsNaN(_prevTs) ? 0.0 : ts - _prevTs;
        if (dt < 0) dt = 0;
        _prevTs = ts;

        if (double.IsNaN(_firstTs)) _firstTs = ts;
        _lastTs = ts;
        _frames++;

        // Possession time, attributed to the controlling team; loose/contested pooled together.
        switch (state.Context.Possession)
        {
            case PossessionState.TeamAControlled: _controlledA += dt; break;
            case PossessionState.TeamBControlled: _controlledB += dt; break;
            default: _loose += dt; break;
        }

        // Physical stats, per player.
        foreach (var p in obs.Players)
        {
            if (!_players.TryGetValue(p.PlayerId, out var acc))
                _players[p.PlayerId] = acc = new PlayerAccum(p.PlayerId, p.Team);
            acc.Team = p.Team; // team can resolve later in the match
            acc.TimeOnPitch += dt;

            if (acc.HasPrev && dt > 0)
            {
                double dx = p.X - acc.PrevX, dy = p.Y - acc.PrevY;
                double step = Math.Sqrt(dx * dx + dy * dy);
                double speed = step / dt;
                if (speed <= MaxPlayerSpeed)  // else: teleport / re-track artefact, ignore
                {
                    acc.Distance += step;
                    acc.SpeedTime += dt;
                    if (speed > acc.TopSpeed) acc.TopSpeed = speed;
                    if (speed >= SprintSpeed) acc.SprintDwell += dt;
                    else acc.CloseSprint();
                }
            }
            acc.PrevX = p.X; acc.PrevY = p.Y; acc.HasPrev = true;
        }
    }

    private void Tally(GameEvent e)
    {
        switch (e.Type)
        {
            case GameEventType.Pass:
                if (e.Team == TeamId.A) _passesA++; else if (e.Team == TeamId.B) _passesB++;
                break;
            case GameEventType.WaywardPass:
                if (e.Team == TeamId.A) _turnoversA++; else if (e.Team == TeamId.B) _turnoversB++;
                break;
        }
    }

    private MatchStatsReport Build()
    {
        double controlled = _controlledA + _controlledB;
        double duration = double.IsNaN(_firstTs) ? 0.0 : _lastTs - _firstTs;

        var players = _players.Values
            .Select(a => a.ToStats())
            .OrderByDescending(p => p.DistanceMeters)
            .ToList();

        return new MatchStatsReport(
            _frames, duration, controlled, _loose,
            TeamStatsFor(_controlledA, controlled, _passesA, _turnoversA),
            TeamStatsFor(_controlledB, controlled, _passesB, _turnoversB),
            players);
    }

    private static TeamStats TeamStatsFor(double controlledTeam, double controlledTotal, int passes, int turnovers)
    {
        double possessionPct = controlledTotal > 0 ? 100.0 * controlledTeam / controlledTotal : 0.0;
        int attempts = passes + turnovers;
        double accuracy = attempts > 0 ? 100.0 * passes / attempts : 0.0;
        return new TeamStats(Round(possessionPct), passes, turnovers, Round(accuracy));
    }

    private static double Round(double v) => Math.Round(v, 1);

    /// <summary>Mutable per-player accumulator. PlayerId is the engine identity
    /// (<see cref="PlayerIdentity.FromTrack"/> of the track id); we recover the track id for the
    /// report so it reads in the same terms as the annotated video.</summary>
    private sealed class PlayerAccum
    {
        public PlayerAccum(int playerId, TeamId team) { PlayerId = playerId; Team = team; }

        public int PlayerId;
        public TeamId Team;
        public double Distance;
        public double TopSpeed;
        public double SpeedTime;
        public double TimeOnPitch;
        public int Sprints;
        public double SprintDwell;
        public bool HasPrev;
        public double PrevX, PrevY;

        public void CloseSprint()
        {
            if (SprintDwell >= SprintMinDuration) Sprints++;
            SprintDwell = 0;
        }

        public PlayerStats ToStats()
        {
            CloseSprint(); // count a sprint still open at the last seen frame
            double avgSpeed = SpeedTime > 0 ? Distance / SpeedTime : 0.0;
            int? trackId = PlayerIdentity.IsStable(PlayerId) ? null : -(PlayerId + 2); // inverse of FromTrack
            return new PlayerStats(
                PlayerId,
                trackId,
                Team.ToString(),
                Math.Round(Distance, 1),
                Math.Round(TopSpeed * 3.6, 1),
                Math.Round(avgSpeed * 3.6, 1),
                Sprints,
                Math.Round(TimeOnPitch, 1));
        }
    }
}
