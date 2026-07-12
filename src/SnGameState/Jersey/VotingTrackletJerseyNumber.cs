using TrackLab.Datastruct;
using TrackLab.Pipeline;
using TrackLab.Utils;

namespace SnGameState.Jersey;

/// <summary>Port of sn_gamestate.jersey.voting_tracklet_jn_api.VotingTrackletJerseyNumber:
/// assigns to every detection of a tracklet the confidence-weighted majority jersey number.</summary>
public sealed class VotingTrackletJerseyNumber : VideoLevelModule
{
    public override IReadOnlyDictionary<string, string[]> InputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "track_id", "jersey_number", "jn_confidence" },
    };

    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; } = new Dictionary<string, string[]>
    {
        ["detection"] = new[] { "jn_tracklet" },
    };

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        foreach (var row in detections.Rows)
            row["jn_tracklet"] = null;
        if (!detections.Columns.Contains("track_id"))
            return detections;

        foreach (var tracklet in detections.Rows.Where(r => r.Has("track_id"))
                                                .GroupBy(r => r.Get<long>("track_id")))
        {
            var jerseyNumber = AttributeVoting.SelectHighestVotedAttribute(
                tracklet.Select(r => r.GetOrDefault<string>("jersey_number")),
                tracklet.Select(r => r.GetOrDefault<double>("jn_confidence")));
            foreach (var row in tracklet)
                row["jn_tracklet"] = jerseyNumber;
        }
        return detections;
    }
}
