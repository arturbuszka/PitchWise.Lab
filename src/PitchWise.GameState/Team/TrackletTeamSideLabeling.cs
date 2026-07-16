using PitchWise.GameState.Calibration;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.GameState.Team;

/// <summary>
/// Labels each team as the left/right side:
/// maps team clusters 0/1 to "left"/"right" by comparing the average pitch x-coordinate
/// of each cluster; goalkeepers are labeled by which half of the pitch they stand in.
/// </summary>
public sealed class TrackletTeamSideLabeling : VideoLevelModule
{
    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id", "team_cluster", "bbox_pitch", "role" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "team" },
    };

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        if (!detections.Columns.Contains("track_id"))
            return detections;

        var teamA = detections.Rows.Where(r => Equals(r.GetOrDefault<int?>("team_cluster"), 0)).ToList();
        var teamB = detections.Rows.Where(r => Equals(r.GetOrDefault<int?>("team_cluster"), 1)).ToList();

        var avgA = MeanPitchX(teamA);
        var avgB = MeanPitchX(teamB);

        var (labelA, labelB) = avgA > avgB ? ("right", "left") : ("left", "right");
        foreach (var row in teamA) row["team"] = labelA;
        foreach (var row in teamB) row["team"] = labelB;

        foreach (var goalkeeper in detections.Rows.Where(r =>
                     r.GetOrDefault<string>("role") == "goalkeeper" && r["bbox_pitch"] is BboxPitch))
            goalkeeper["team"] = goalkeeper.Get<BboxPitch>("bbox_pitch").XBottomMiddle > 0 ? "right" : "left";

        return detections;
    }

    private static double MeanPitchX(IEnumerable<Series> rows)
    {
        var xs = rows.Select(r => r.GetOrDefault<BboxPitch>("bbox_pitch"))
                     .Where(b => b is not null)
                     .Select(b => b!.XBottomMiddle)
                     .ToList();
        return xs.Count == 0 ? double.NaN : xs.Average();
    }
}
