using CommunityToolkit.Mvvm.ComponentModel;
using PitchWise.GameState.WorldState;

namespace PitchWise.App.Core.ViewModels;

/// <summary>The match statistics panel.</summary>
/// <remarks>
/// Bound to a report held here rather than read from stats.json on demand, because the tuning
/// panel recomputes it live: the same view has to show whatever the current EngineConfig says.
/// </remarks>
public sealed partial class StatsViewModel : ViewModelBase
{
    public StatsViewModel(MatchStatsReport report) => _report = report;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(TeamA))]
    [NotifyPropertyChangedFor(nameof(TeamB))]
    [NotifyPropertyChangedFor(nameof(Players))]
    [NotifyPropertyChangedFor(nameof(DurationText))]
    [NotifyPropertyChangedFor(nameof(BallStateText))]
    [NotifyPropertyChangedFor(nameof(PossessionABar))]
    [NotifyPropertyChangedFor(nameof(PossessionBBar))]
    private MatchStatsReport _report;

    public TeamStats TeamA => Report.TeamA;

    public TeamStats TeamB => Report.TeamB;

    /// <summary>Players by distance covered — the ordering anyone opening this panel wants first.</summary>
    public IReadOnlyList<PlayerStats> Players =>
        Report.Players.OrderByDescending(p => p.DistanceMeters).ToList();

    public string DurationText =>
        $"{TimeSpan.FromSeconds(Report.DurationSeconds):mm\\:ss} · {Report.FramesProcessed} frames";

    /// <summary>
    /// Possession splits controlled time only, so loose-ball time is worth showing beside it —
    /// a high loose figure is the tell that the ball was rarely tracked.
    /// </summary>
    public string BallStateText =>
        $"controlled {Report.ControlledSeconds:F1}s · loose {Report.LooseSeconds:F1}s";

    // Star sizes for the possession bar: A + B = 100 over controlled time, so these are already
    // the proportions to hand a Grid.
    public double PossessionABar => Math.Max(TeamA.PossessionPct, 0.001);
    public double PossessionBBar => Math.Max(TeamB.PossessionPct, 0.001);
}
