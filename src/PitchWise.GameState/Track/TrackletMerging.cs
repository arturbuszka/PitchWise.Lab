using Microsoft.Extensions.Logging;
using PitchWise.GameState.Calibration;
using PitchWise.GameState.Team;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Eval;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Track;

/// <summary>Settings of <see cref="TrackletMerging"/>; the defaults are those of the Python prototype
/// (docs/eval-baseline.md, "Prototyp sklejania offline").</summary>
public sealed record TrackletMergingOptions
{
    /// <summary>Longest time between the end of one tracklet and the start of the next (seconds).</summary>
    public double MaxGapSeconds { get; init; } = 10;
    /// <summary>Pitch distance allowed regardless of the gap (meters): calibration noise.</summary>
    public double MaxDistanceMeters { get; init; } = 3;
    /// <summary>Pitch distance allowed per second of gap (m/s).</summary>
    public double MaxSpeed { get; init; } = 8;
    /// <summary>Largest cosine distance between the tracklets' mean ReID embeddings.</summary>
    public double MaxAppearanceDistance { get; init; } = 0.3;
    /// <summary>Weight of the distance term (pitch distance / reachable distance) in the link cost.</summary>
    public double DistanceWeight { get; init; } = 0.3;
    /// <summary>Weight of the gap term (gap / <see cref="MaxGapSeconds"/>) in the link cost.</summary>
    public double GapWeight { get; init; } = 0.1;
    /// <summary>Cost of leaving a tracklet end or start unlinked: a link costing more is never made.</summary>
    public double NoLinkCost { get; init; } = 0.1;
    /// <summary>Tracklets with fewer detections get no team (their jersey colour is unreliable), so
    /// the team constraint does not apply to them.</summary>
    public int MinTeamDetections { get; init; } = 10;
}

/// <summary>
/// Offline tracklet merging: joins a tracklet that ends to one that starts later and continues the
/// same person, typically after the person left the frame or was occluded for longer than the
/// tracker's Kalman prediction can bridge. Not part of the Python sn-gamestate pipeline.
/// </summary>
/// <remarks>
/// One min-cost assignment of tracklet ends to tracklet starts (each tracklet gets at most one
/// successor and one predecessor), so chains of fragments collapse into one track id, the id of the
/// chain's first tracklet. A link needs:
/// <list type="bullet">
/// <item>tracklets disjoint in time, the second starting at most <see cref="TrackletMergingOptions.MaxGapSeconds"/> after the first ends;</item>
/// <item>pitch distance between the end and the start (median over 5 detections) of at most
/// <see cref="TrackletMergingOptions.MaxDistanceMeters"/> + <see cref="TrackletMergingOptions.MaxSpeed"/>·gap,
/// unless either position is unknown;</item>
/// <item>the same team when both tracklets have a reliable one: a player team from KMeans (k=2) over
/// the standardised median jersey colour (<see cref="TorsoColor"/>), or the goalkeeper / referee role;</item>
/// <item>cosine distance of the mean ReID embeddings of at most <see cref="TrackletMergingOptions.MaxAppearanceDistance"/>.</item>
/// </list>
/// Cost: appearance + DistanceWeight·distance/reachable + GapWeight·min(gap/MaxGap, 1); leaving an
/// end or start unlinked costs <see cref="TrackletMergingOptions.NoLinkCost"/>.
/// Runs before the tracklet-level voting, so roles come from the per-detection role_detection.
/// </remarks>
public sealed class TrackletMerging : VideoLevelModule
{
    /// <summary>Detections at each tracklet end whose pitch positions are medianed.</summary>
    private const int EndpointDetections = 5;
    private const int MinTeamClusterTracklets = 2;

    private readonly TrackletMergingOptions _options;
    private readonly double _fps;
    private readonly ILogger? _log;

    /// <param name="fps">Frame rate, to turn frame gaps into seconds; 25 (SoccerNet) when unknown.</param>
    public TrackletMerging(TrackletMergingOptions? options = null, double fps = 25, ILogger? log = null)
    {
        _options = options ?? new TrackletMergingOptions();
        _fps = fps > 0 ? fps : 25;
        _log = log;
    }

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "image_id", "track_id", "bbox_pitch", "embeddings", "torso_color", "role_detection" },
        ["image"] = new[] { "frame" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id" },
    };

    /// <summary>Number of links made in the last <see cref="Process"/> call.</summary>
    public int LastLinkCount { get; private set; }

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        LastLinkCount = 0;
        if (!detections.Columns.Contains("track_id"))
            return detections;

        var frameOf = imageMetadatas.Rows.ToDictionary(r => r.Index, r => r.Get<long>("frame"));
        var tracklets = detections.Rows
            .Where(r => r.Has("track_id"))
            .GroupBy(r => r.Get<long>("track_id"))
            .OrderBy(g => g.Key)
            .Select(g => Tracklet.From(g.Key, g.Select(r => (Frame: frameOf[r.Get<long>("image_id")], Row: r))
                                                  .OrderBy(x => x.Frame).ToList()))
            .ToList();
        AssignTeams(tracklets);

        var successor = Link(tracklets);
        LastLinkCount = successor.Count;
        _log?.LogInformation("TrackletMerging: {Links} links over {Tracklets} tracklets", successor.Count, tracklets.Count);
        if (successor.Count == 0)
            return detections;

        var newId = new Dictionary<long, long>();
        foreach (var head in tracklets.Where(t => !successor.ContainsValue(t)))
            for (var t = head; successor.TryGetValue(t, out var next); t = next)
                newId[next.Id] = head.Id;
        foreach (var row in detections.Rows)
            if (row.Has("track_id") && newId.TryGetValue(row.Get<long>("track_id"), out var id))
                row["track_id"] = id;
        return detections;
    }

    /// <summary>Tracklet → its successor, from one assignment of ends to starts.</summary>
    internal Dictionary<Tracklet, Tracklet> Link(IReadOnlyList<Tracklet> tracklets)
    {
        var n = tracklets.Count;
        var cost = new float[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                cost[i, j] = (float)LinkCost(tracklets[i], tracklets[j]);

        // "no link" as an assignment option: each end i may take dummy start n+i and each start j
        // dummy end n+j at NoLinkCost; dummy ends and dummy starts pair up for free
        var m = 2 * n;
        var augmented = new float[m, m];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
                augmented[i, j] = (i < n, j < n) switch
                {
                    (true, true) => cost[i, j],
                    (false, false) => 0f,
                    _ => i % n == j % n ? (float)_options.NoLinkCost : float.PositiveInfinity,
                };

        var assignment = Hungarian.Solve(augmented);
        var successor = new Dictionary<Tracklet, Tracklet>();
        for (var i = 0; i < n; i++)
            if (assignment[i] is var j and >= 0 && j < n && !float.IsInfinity(cost[i, j]))
                successor[tracklets[i]] = tracklets[j];
        return successor;
    }

    /// <summary>Cost of continuing <paramref name="a"/> with <paramref name="b"/>, +∞ when not allowed.</summary>
    internal double LinkCost(Tracklet a, Tracklet b)
    {
        var maxGapFrames = (long)Math.Round(_options.MaxGapSeconds * _fps);
        // gap = frames in between without either tracklet
        if (a == b || b.Start <= a.End || b.Start - a.End - 1 > maxGapFrames)
            return double.PositiveInfinity;
        if (a.Team is not null && b.Team is not null && a.Team != b.Team)
            return double.PositiveInfinity;

        var gapSeconds = (b.Start - a.End) / _fps;
        var reach = _options.MaxDistanceMeters + _options.MaxSpeed * gapSeconds;
        var distance = Math.Sqrt(Math.Pow(b.StartX - a.EndX, 2) + Math.Pow(b.StartY - a.EndY, 2));
        if (distance > reach)   // false for NaN: unknown positions do not constrain
            return double.PositiveInfinity;

        if (a.MeanEmbedding is null || b.MeanEmbedding is null)
            return double.PositiveInfinity;
        var appearance = 1 - Dot(a.MeanEmbedding, b.MeanEmbedding);
        if (appearance > _options.MaxAppearanceDistance)
            return double.PositiveInfinity;

        return appearance
               + _options.DistanceWeight * (double.IsNaN(distance) ? 0 : distance / reach)
               + _options.GapWeight * Math.Min(gapSeconds / _options.MaxGapSeconds, 1);
    }

    /// <summary>Player tracklets with a reliable colour get team 0/1 from KMeans over the
    /// standardised colours; goalkeepers and referees are their own teams.</summary>
    private void AssignTeams(IReadOnlyList<Tracklet> tracklets)
    {
        var reliable = tracklets.Where(t => t.Detections >= _options.MinTeamDetections).ToList();
        foreach (var t in reliable.Where(t => t.Role is "goalkeeper" or "referee"))
            t.Team = t.Role;

        var players = reliable.Where(t => t.Role == "player" && t.Color is not null).ToList();
        if (players.Count < MinTeamClusterTracklets)
        {
            // nothing to split, but still not a goalkeeper or referee
            foreach (var t in players) t.Team = "team0";
            return;
        }
        var points = players.Select(t => t.Color!.ToArray()).ToArray();
        for (var d = 0; d < points[0].Length; d++)
        {
            var mean = points.Average(p => (double)p[d]);
            var std = Math.Sqrt(points.Average(p => (p[d] - mean) * (p[d] - mean))) + 1e-9;
            foreach (var p in points)
                p[d] = (float)((p[d] - mean) / std);
        }
        var labels = KMeans.Fit(points, k: 2, seed: 0, restarts: 10);
        for (var i = 0; i < players.Count; i++)
            players[i].Team = $"team{labels[i]}";
    }

    private static double Dot(double[] a, double[] b)
    {
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal sealed class Tracklet
    {
        public required long Id { get; init; }
        public required long Start { get; init; }
        public required long End { get; init; }
        public required int Detections { get; init; }
        /// <summary>Median pitch position over the first / last detections; NaN when none is known.</summary>
        public double StartX { get; init; } = double.NaN;
        public double StartY { get; init; } = double.NaN;
        public double EndX { get; init; } = double.NaN;
        public double EndY { get; init; } = double.NaN;
        /// <summary>Unit-length mean of the unit-length detection embeddings.</summary>
        public double[]? MeanEmbedding { get; init; }
        /// <summary>Per-channel median jersey colour over the detections that have one.</summary>
        public float[]? Color { get; init; }
        /// <summary>Most frequent role_detection (ties: the one seen first).</summary>
        public string? Role { get; init; }
        /// <summary>Set by <see cref="AssignTeams"/>: null when unreliable.</summary>
        public string? Team { get; set; }

        /// <param name="detections">The tracklet's detections in frame order.</param>
        public static Tracklet From(long id, IReadOnlyList<(long Frame, Series Row)> detections)
        {
            var rows = detections.Select(d => d.Row).ToList();
            var (startX, startY) = MedianPosition(rows.Take(EndpointDetections));
            var (endX, endY) = MedianPosition(rows.TakeLast(EndpointDetections));
            var colors = rows.Select(r => r.GetOrDefault<float[]>("torso_color")).OfType<float[]>().ToList();
            return new Tracklet
            {
                Id = id,
                Start = detections[0].Frame,
                End = detections[^1].Frame,
                Detections = rows.Count,
                StartX = startX, StartY = startY, EndX = endX, EndY = endY,
                MeanEmbedding = UnitMean(rows.Select(r => r.GetOrDefault<float[]>("embeddings")).OfType<float[]>()),
                Color = colors.Count == 0 ? null
                    : Enumerable.Range(0, colors[0].Length).Select(c => TorsoColor.Median(colors.Select(x => x[c]).ToList())).ToArray(),
                Role = rows.Select(r => r.GetOrDefault<string>("role_detection")).OfType<string>()
                    .Select((role, order) => (role, order))
                    .GroupBy(x => x.role)
                    .OrderByDescending(g => g.Count()).ThenBy(g => g.Min(x => x.order))
                    .Select(g => g.Key).FirstOrDefault(),
            };
        }

        private static (double X, double Y) MedianPosition(IEnumerable<Series> rows)
        {
            var positions = rows.Select(r => r.GetOrDefault<BboxPitch>("bbox_pitch")).OfType<BboxPitch>().ToList();
            return positions.Count == 0
                ? (double.NaN, double.NaN)
                : (Median(positions.Select(p => p.XBottomMiddle)), Median(positions.Select(p => p.YBottomMiddle)));
        }

        private static double Median(IEnumerable<double> values)
        {
            var sorted = values.Order().ToList();
            var mid = sorted.Count / 2;
            return sorted.Count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        }

        private static double[]? UnitMean(IEnumerable<float[]> embeddings)
        {
            double[]? sum = null;
            foreach (var e in embeddings)
            {
                sum ??= new double[e.Length];
                var norm = Norm(e.Select(x => (double)x));
                for (var i = 0; i < e.Length; i++) sum[i] += norm > 1e-12 ? e[i] / norm : e[i];
            }
            if (sum is null)
                return null;
            var total = Norm(sum);
            return total > 1e-12 ? sum.Select(x => x / total).ToArray() : sum;
        }

        private static double Norm(IEnumerable<double> v) => Math.Sqrt(v.Sum(x => x * x));
    }
}
