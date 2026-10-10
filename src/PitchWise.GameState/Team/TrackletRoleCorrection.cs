using Microsoft.Extensions.Logging;
using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Team;

/// <summary>Settings of <see cref="TrackletRoleCorrection"/>; the defaults are those of the Python
/// prototype (docs/eval-baseline.md, "Korekta ról po kolorze koszulki").</summary>
public sealed record TrackletRoleCorrectionOptions
{
    /// <summary>More "referee" detections per frame on average than this and the referee group is a team.</summary>
    public double MaxRefereesPerFrame { get; init; } = 3;
    /// <summary>More "goalkeeper" detections per frame on average than this and the goalkeeper group is a team.</summary>
    public double MaxGoalkeepersPerFrame { get; init; } = 2;
    /// <summary>Weight of the Lab lightness in colour distances: shadow changes L far more than a, b.</summary>
    public double LightnessWeight { get; init; } = 0.25;
    /// <summary>When a crowded group is a team: tracklets closer than this (weighted Lab) to either
    /// team colour are players, "player" tracklets farther from both are not.</summary>
    public double TeamColorRadius { get; init; } = 20;
    /// <summary>A goalkeeper's colour is farther than this (weighted Lab) from both team colours.</summary>
    public double GoalkeeperColorDistance { get; init; } = 15;
    /// <summary>A goalkeeper's mean pitch |x| is beyond this (meters; the penalty area starts at 36).</summary>
    public double GoalkeeperMinAbsX { get; init; } = 38;
    /// <summary>Tracklets with fewer coloured detections keep their role.</summary>
    public int MinColorDetections { get; init; } = 10;
}

/// <summary>
/// Corrects the tracklet roles voted from PRTReid with jersey colour. Not part of the Python
/// sn-gamestate pipeline.
/// </summary>
/// <remarks>
/// PRTReid sometimes gives "referee" (partly "goalkeeper") to a whole team, the real referees then
/// "player", and the goalkeeper often "player" (docs/eval-baseline.md, "Diagnoza: niski wariant „bez
/// numerów”"). Two steps over tracklet colours (per-channel median <see cref="TorsoColor"/>, L scaled
/// by <see cref="TrackletRoleCorrectionOptions.LightnessWeight"/>):
/// <list type="number">
/// <item>Officials are at most a few at a time. A referee (goalkeeper) group with more detections per
/// frame than <see cref="TrackletRoleCorrectionOptions.MaxRefereesPerFrame"/>
/// (<see cref="TrackletRoleCorrectionOptions.MaxGoalkeepersPerFrame"/>) is a team: its median colour
/// and that of the player group are the two team colours. Tracklets within
/// <see cref="TrackletRoleCorrectionOptions.TeamColorRadius"/> of either become players; "player"
/// tracklets farther from both are outsiders.</item>
/// <item>Team colours = cluster medians of KMeans (k=2) over the player tracklets. A player or
/// outsider tracklet farther than <see cref="TrackletRoleCorrectionOptions.GoalkeeperColorDistance"/>
/// from both, with mean pitch |x| beyond <see cref="TrackletRoleCorrectionOptions.GoalkeeperMinAbsX"/>,
/// is a goalkeeper: one per half, the farthest in colour. The remaining outsiders are referees.</item>
/// </list>
/// Runs after the tracklet voting and before <see cref="TrackletTeamClustering"/>, which only
/// clusters "player" tracklets.
/// </remarks>
public sealed class TrackletRoleCorrection : VideoLevelModule
{
    private const int MinGoalkeeperClusterTracklets = 4;

    private readonly TrackletRoleCorrectionOptions _options;
    private readonly ILogger? _log;

    public TrackletRoleCorrection(TrackletRoleCorrectionOptions? options = null, ILogger? log = null)
    {
        _options = options ?? new TrackletRoleCorrectionOptions();
        _log = log;
    }

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "image_id", "track_id", "role", "torso_color", "bbox_pitch" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "role" },
    };

    /// <summary>Tracklets whose role changed in the last <see cref="Process"/> call.</summary>
    public int LastChangedCount { get; private set; }

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        LastChangedCount = 0;
        if (!detections.Columns.Contains("track_id"))
            return detections;

        var tracked = detections.Rows.Where(r => r.Has("track_id")).ToList();
        var frames = Math.Max(tracked.Select(r => r.Get<long>("image_id")).Distinct().Count(), 1);
        double PerFrame(string role) => tracked.Count(r => r.GetOrDefault<string>("role") == role) / (double)frames;
        var crowded = new List<string>();
        if (PerFrame("referee") > _options.MaxRefereesPerFrame) crowded.Add("referee");
        if (PerFrame("goalkeeper") > _options.MaxGoalkeepersPerFrame) crowded.Add("goalkeeper");

        var tracklets = tracked
            .GroupBy(r => r.Get<long>("track_id"))
            .OrderBy(g => g.Key)
            .Select(g => Tracklet.From(g.ToList(), _options.LightnessWeight))
            .Where(t => t.ColorDetections >= _options.MinColorDetections && t.VotedRole is not null)
            .ToList();
        foreach (var t in tracklets)
            t.Role = t.VotedRole;

        if (crowded.Count > 0)
            ReclaimCrowdedTeam(tracklets, crowded);
        FindGoalkeepers(tracklets);
        foreach (var t in tracklets.Where(t => t.Outsider))
            t.Role = "referee";

        foreach (var t in tracklets.Where(t => t.Role != t.VotedRole))
        {
            foreach (var row in t.Rows)
                row["role"] = t.Role;
            LastChangedCount++;
        }
        _log?.LogInformation("TrackletRoleCorrection: crowded groups [{Crowded}], {Changed} of {Tracklets} tracklets changed role",
            string.Join(", ", crowded), LastChangedCount, tracklets.Count);
        return detections;
    }

    /// <summary>Step 1: the crowded group is a team (see the class remarks).</summary>
    private void ReclaimCrowdedTeam(IReadOnlyList<Tracklet> tracklets, IReadOnlyCollection<string> crowded)
    {
        var crowd = tracklets.Where(t => crowded.Contains(t.VotedRole!)).Select(t => t.Color).ToList();
        var players = tracklets.Where(t => t.VotedRole == "player").Select(t => t.Color).ToList();
        if (crowd.Count == 0 || players.Count == 0)
            return;
        var teamColors = new[] { Median(crowd), Median(players) };
        foreach (var t in tracklets)
        {
            if (Nearest(t.Color, teamColors) <= _options.TeamColorRadius)
                t.Role = "player";
            else if (t.VotedRole == "player")
                t.Outsider = true;
        }
    }

    /// <summary>Step 2: goalkeepers among the player and outsider tracklets (see the class remarks).</summary>
    private void FindGoalkeepers(IReadOnlyList<Tracklet> tracklets)
    {
        var players = tracklets.Where(t => t.Role == "player" && !t.Outsider).ToList();
        if (players.Count < MinGoalkeeperClusterTracklets)
            return;
        var points = players.Select(t => t.Color.ToArray()).ToArray();
        Standardize(points);
        var labels = KMeans.Fit(points, k: 2, seed: 0, restarts: 10);
        var teamColors = Enumerable.Range(0, 2)
            .Select(k => players.Where((_, i) => labels[i] == k).Select(t => t.Color).ToList())
            .Where(c => c.Count > 0)
            .Select(Median)
            .ToArray();

        var best = new Dictionary<bool, (double Distance, Tracklet Tracklet)>();
        foreach (var t in tracklets.Where(t => (t.Role == "player" || t.Outsider) && !double.IsNaN(t.MeanX)))
        {
            var distance = Nearest(t.Color, teamColors);
            var right = t.MeanX > 0;
            if (distance > _options.GoalkeeperColorDistance && Math.Abs(t.MeanX) > _options.GoalkeeperMinAbsX
                && (!best.TryGetValue(right, out var b) || distance > b.Distance))
                best[right] = (distance, t);
        }
        foreach (var (_, t) in best.Values)
        {
            t.Role = "goalkeeper";
            t.Outsider = false;
        }
    }

    private static double Nearest(float[] color, IEnumerable<float[]> centers)
        => centers.Min(c => Math.Sqrt(c.Zip(color, (a, b) => (double)(a - b) * (a - b)).Sum()));

    /// <summary>Per-channel median (numpy.median) of several colours.</summary>
    private static float[] Median(IReadOnlyList<float[]> colors)
        => Enumerable.Range(0, colors[0].Length).Select(d => TorsoColor.Median(colors.Select(c => c[d]).ToList())).ToArray();

    /// <summary>Zero mean, unit (population) standard deviation per dimension, in place.</summary>
    private static void Standardize(float[][] points)
    {
        for (var d = 0; d < points[0].Length; d++)
        {
            var mean = points.Average(p => (double)p[d]);
            var std = Math.Sqrt(points.Average(p => (p[d] - mean) * (p[d] - mean))) + 1e-9;
            foreach (var p in points)
                p[d] = (float)((p[d] - mean) / std);
        }
    }

    private sealed class Tracklet
    {
        public required IReadOnlyList<Series> Rows { get; init; }
        public required string? VotedRole { get; init; }
        /// <summary>Median torso colour with L scaled by the lightness weight.</summary>
        public required float[] Color { get; init; }
        public required int ColorDetections { get; init; }
        /// <summary>Mean pitch x of the detections that have one; NaN without any.</summary>
        public required double MeanX { get; init; }
        public string? Role { get; set; }
        /// <summary>Voted "player" but far from both team colours: a referee unless it is a goalkeeper.</summary>
        public bool Outsider { get; set; }

        public static Tracklet From(IReadOnlyList<Series> rows, double lightnessWeight)
        {
            var colors = rows.Select(r => r.GetOrDefault<float[]>("torso_color")).OfType<float[]>().ToList();
            var color = colors.Count == 0
                ? Array.Empty<float>()
                : Enumerable.Range(0, colors[0].Length).Select(d => TorsoColor.Median(colors.Select(c => c[d]).ToList())).ToArray();
            if (color.Length > 0)
                color[0] = (float)(color[0] * lightnessWeight);
            var xs = rows.Select(r => r.GetOrDefault<BboxPitch>("bbox_pitch")).OfType<BboxPitch>().Select(b => b.XBottomMiddle).ToList();
            return new Tracklet
            {
                Rows = rows,
                VotedRole = rows.Select(r => r.GetOrDefault<string>("role")).FirstOrDefault(r => r is not null),
                Color = color,
                ColorDetections = colors.Count,
                MeanX = xs.Count == 0 ? double.NaN : xs.Average(),
            };
        }
    }
}
