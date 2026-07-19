using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.ViewModels;

/// <summary>
/// One track's life, summarised: what the pipeline decided about it and where to look.
/// </summary>
/// <remarks>
/// The per-detection attributes are majority-voted across the tracklet by MajorityVoteTracklet,
/// so a summary is the honest unit here — a single frame's role or jersey is noise.
/// </remarks>
public sealed record TrackletSummary(
    long TrackId,
    string Team,
    string Role,
    string Jersey,
    int Detections,
    int FirstFrame,
    int LastFrame)
{
    public string Span => $"{FirstFrame}–{LastFrame}";

    public string Label => Jersey is { Length: > 0 } and not "?"
        ? $"#{TrackId} ({Jersey})"
        : $"#{TrackId}";

    /// <summary>Groups detections into one row per track.</summary>
    /// <param name="frameOf">Maps an image_id to its frame index, for the seek target.</param>
    public static IReadOnlyList<TrackletSummary> From(DataFrame detections, Func<long, int> frameOf)
    {
        var tracklets = new List<TrackletSummary>();

        foreach (var group in detections.Rows
            .Where(d => d.Has("track_id"))
            .GroupBy(d => d.Get<long>("track_id")))
        {
            var frames = group
                .Select(d => frameOf(d.Get<long>("image_id")))
                .Where(f => f >= 0)
                .ToList();
            if (frames.Count == 0) continue;

            tracklets.Add(new TrackletSummary(
                group.Key,
                Majority(group, "team"),
                Majority(group, "role"),
                Majority(group, "jersey_number"),
                group.Count(),
                frames.Min(),
                frames.Max()));
        }

        return tracklets.OrderBy(t => t.FirstFrame).ThenBy(t => t.TrackId).ToList();
    }

    private static string Majority(IEnumerable<Series> rows, string column)
    {
        var value = rows
            .Select(r => r.GetOrDefault<string>(column))
            .Where(v => !string.IsNullOrEmpty(v))
            .GroupBy(v => v!)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault()?.Key;
        return value ?? "?";
    }
}
