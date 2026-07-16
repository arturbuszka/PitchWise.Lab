using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Team;

/// <summary>Clusters tracklets into two teams:
/// KMeans (k=2) over mean ReID embeddings of "player" tracklets to split them into two teams.
/// Cluster ids 0/1 are mapped to left/right by <c>TrackletTeamSideLabeling</c>.</summary>
public sealed class TrackletTeamClustering : VideoLevelModule
{
    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id", "embeddings", "role" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "team_cluster" },
    };

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        var trackletEmbeddings = detections.Rows
            .Where(r => r.GetOrDefault<string>("role") == "player" && r.Has("track_id") && r.Has("embeddings"))
            .GroupBy(r => r.Get<long>("track_id"))
            .Select(g => (TrackId: g.Key, Embedding: MeanEmbedding(g.Select(r => r.Get<float[]>("embeddings")))))
            .ToList();

        if (trackletEmbeddings.Count == 0)
        {
            foreach (var row in detections.Rows)
                row["team_cluster"] = null;
            return detections;
        }

        Dictionary<long, int> clusterByTrack;
        if (trackletEmbeddings.Count == 1)
        {
            clusterByTrack = new Dictionary<long, int> { [trackletEmbeddings[0].TrackId] = 0 };
        }
        else
        {
            var labels = KMeans.Fit(trackletEmbeddings.Select(t => t.Embedding).ToArray(), k: 2, seed: 0);
            clusterByTrack = trackletEmbeddings
                .Select((t, i) => (t.TrackId, Label: labels[i]))
                .ToDictionary(x => x.TrackId, x => x.Label);
        }

        foreach (var row in detections.Rows)
            row["team_cluster"] = row.Has("track_id") && clusterByTrack.TryGetValue(row.Get<long>("track_id"), out var c)
                ? c : null;
        return detections;
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

/// <summary>Minimal KMeans (Lloyd's algorithm, k-means++ init) replacing sklearn.cluster.KMeans.</summary>
internal static class KMeans
{
    public static int[] Fit(float[][] points, int k, int seed, int maxIter = 300)
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
        return labels;
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
