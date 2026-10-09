using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Team;

/// <summary>Clusters tracklets into two teams:
/// KMeans (k=2) over one feature per "player" tracklet to split them into two teams.
/// Cluster ids 0/1 are mapped to left/right by <c>TrackletTeamSideLabeling</c>.</summary>
/// <remarks>
/// The Python sn-gamestate module clusters mean ReID embeddings (<see cref="TeamFeature.Embedding"/>).
/// <see cref="TeamFeature.Color"/> clusters the jersey colour from <see cref="TorsoColor"/> instead:
/// the per-channel median over the tracklet, standardised across tracklets. Tracklets without any
/// colour get no cluster.
/// </remarks>
public sealed class TrackletTeamClustering : VideoLevelModule
{
    /// <summary>KMeans restarts for <see cref="TeamFeature.Color"/>, best inertia wins (the sklearn
    /// default before 1.4); a single start sometimes lands in a split with a few outlier tracklets
    /// alone in a cluster. <see cref="TeamFeature.Embedding"/> keeps the single start of the port.</summary>
    private const int ColorRestarts = 10;

    private readonly TeamFeature _feature;

    public TrackletTeamClustering(TeamFeature feature = TeamFeature.Embedding)
    {
        _feature = feature;
        InputColumns = new Dictionary<string, string[]>
        {
            ["detection"] = new[] { "track_id", FeatureColumn, "role" },
        };
    }

    private string FeatureColumn => _feature == TeamFeature.Color ? "torso_color" : "embeddings";

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; }

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "team_cluster" },
    };

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        var tracklets = detections.Rows
            .Where(r => r.GetOrDefault<string>("role") == "player" && r.Has("track_id") && r.Has(FeatureColumn))
            .GroupBy(r => r.Get<long>("track_id"))
            .Select(g => (TrackId: g.Key, Feature: Aggregate(g.Select(r => r.Get<float[]>(FeatureColumn)).ToList())))
            .ToList();

        if (tracklets.Count == 0)
        {
            foreach (var row in detections.Rows)
                row["team_cluster"] = null;
            return detections;
        }

        Dictionary<long, int> clusterByTrack;
        if (tracklets.Count == 1)
        {
            clusterByTrack = new Dictionary<long, int> { [tracklets[0].TrackId] = 0 };
        }
        else
        {
            var points = tracklets.Select(t => t.Feature).ToArray();
            if (_feature == TeamFeature.Color)
                Standardize(points);
            var labels = KMeans.Fit(points, k: 2, seed: 0, restarts: _feature == TeamFeature.Color ? ColorRestarts : 1);
            clusterByTrack = tracklets
                .Select((t, i) => (t.TrackId, Label: labels[i]))
                .ToDictionary(x => x.TrackId, x => x.Label);
        }

        foreach (var row in detections.Rows)
            row["team_cluster"] = row.Has("track_id") && clusterByTrack.TryGetValue(row.Get<long>("track_id"), out var c)
                ? c : null;
        return detections;
    }

    private float[] Aggregate(IReadOnlyList<float[]> features)
        => _feature == TeamFeature.Color
            ? Enumerable.Range(0, features[0].Length)
                .Select(d => TorsoColor.Median(features.Select(f => f[d]).ToList()))
                .ToArray()
            : MeanEmbedding(features);

    /// <summary>Zero mean, unit (population) standard deviation per dimension, in place.</summary>
    private static void Standardize(float[][] points)
    {
        for (var d = 0; d < points[0].Length; d++)
        {
            var mean = points.Average(p => (double)p[d]);
            var std = Math.Sqrt(points.Average(p => (p[d] - mean) * (p[d] - mean)));
            foreach (var p in points)
                p[d] = (float)((p[d] - mean) / (std + 1e-9));
        }
    }

    private static float[] MeanEmbedding(IEnumerable<float[]> embeddings)
    {
        float[]? sum = null;
        var n = 0;
        foreach (var e in embeddings)
        {
            sum ??= new float[e.Length];
            for (var i = 0; i < e.Length; i++) sum[i] += e[i];
            n++;
        }
        for (var i = 0; i < sum!.Length; i++) sum[i] /= n;
        return sum;
    }
}

/// <summary>What <see cref="TrackletTeamClustering"/> splits tracklets by.</summary>
public enum TeamFeature
{
    /// <summary>Mean ReID embedding, as the Python sn-gamestate module.</summary>
    Embedding,
    /// <summary>Median torso colour from <see cref="TorsoColor"/>.</summary>
    Color,
}

/// <summary>Minimal KMeans (Lloyd's algorithm, k-means++ init) replacing sklearn.cluster.KMeans.</summary>
internal static class KMeans
{
    /// <summary>Best of <paramref name="restarts"/> runs (seeds <paramref name="seed"/>, seed+1, …) by inertia.</summary>
    public static int[] Fit(float[][] points, int k, int seed, int restarts, int maxIter = 300)
    {
        int[]? best = null;
        var bestInertia = double.MaxValue;
        for (var r = 0; r < restarts; r++)
        {
            var (labels, inertia) = FitOnce(points, k, seed + r, maxIter);
            if (inertia < bestInertia) { bestInertia = inertia; best = labels; }
        }
        return best!;
    }

    private static (int[] Labels, double Inertia) FitOnce(float[][] points, int k, int seed, int maxIter)
    {
        var rng = new Random(seed);
        var centroids = InitPlusPlus(points, k, rng);
        var labels = new int[points.Length];

        for (var iter = 0; iter < maxIter; iter++)
        {
            var changed = false;
            for (var i = 0; i < points.Length; i++)
            {
                var best = 0;
                var bestDist = double.MaxValue;
                for (var c = 0; c < k; c++)
                {
                    var d = SquaredDistance(points[i], centroids[c]);
                    if (d < bestDist) { bestDist = d; best = c; }
                }
                if (labels[i] != best) { labels[i] = best; changed = true; }
            }
            if (!changed && iter > 0) break;

            for (var c = 0; c < k; c++)
            {
                var members = Enumerable.Range(0, points.Length).Where(i => labels[i] == c).ToList();
                if (members.Count == 0) continue;
                var centroid = new float[points[0].Length];
                foreach (var i in members)
                    for (var d = 0; d < centroid.Length; d++) centroid[d] += points[i][d];
                for (var d = 0; d < centroid.Length; d++) centroid[d] /= members.Count;
                centroids[c] = centroid;
            }
        }
        var inertia = Enumerable.Range(0, points.Length).Sum(i => SquaredDistance(points[i], centroids[labels[i]]));
        return (labels, inertia);
    }

    private static float[][] InitPlusPlus(float[][] points, int k, Random rng)
    {
        var centroids = new float[k][];
        centroids[0] = points[rng.Next(points.Length)];
        for (var c = 1; c < k; c++)
        {
            var distances = points.Select(p =>
                Enumerable.Range(0, c).Min(i => SquaredDistance(p, centroids[i]))).ToArray();
            var total = distances.Sum();
            var target = rng.NextDouble() * total;
            var cumulative = 0.0;
            var chosen = points.Length - 1;
            for (var i = 0; i < points.Length; i++)
            {
                cumulative += distances[i];
                if (cumulative >= target) { chosen = i; break; }
            }
            centroids[c] = points[chosen];
        }
        return centroids;
    }

    private static double SquaredDistance(float[] a, float[] b)
    {
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            var d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }
}
