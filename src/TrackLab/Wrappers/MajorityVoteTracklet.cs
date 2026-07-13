using TrackLab.Datastruct;
using TrackLab.Pipeline;
using TrackLab.Utils;

namespace TrackLab.Wrappers;

/// <summary>
/// Port of tracklab.wrappers.tracklet_agg.majority_vote_api.MajorityVoteTracklet:
/// for each configured attribute, aggregates per-detection predictions
/// ("{attribute}_detection" weighted by "{attribute}_confidence") into a single
/// tracklet-level value written to "{attribute}". The SoccerNet baseline uses
/// attributes = ["jersey_number", "role"].
/// </summary>
public sealed class MajorityVoteTracklet : VideoLevelModule
{
    private readonly string[] _attributes;

    public override IReadOnlyDictionary<string, string[]> InputColumns { get; }
    public override IReadOnlyDictionary<string, string[]> OutputColumns { get; }

    public MajorityVoteTracklet(params string[] attributes)
    {
        _attributes = attributes;
        InputColumns = new Dictionary<string, string[]>
        {
            ["detection"] = attributes
                .SelectMany(a => new[] { $"{a}_detection", $"{a}_confidence" })
                .Append("track_id")
                .ToArray(),
        };
        OutputColumns = new Dictionary<string, string[]> { ["detection"] = attributes };
    }

    public override DataFrame Process(DataFrame detections, DataFrame imageMetadatas)
    {
        foreach (var row in detections.Rows)
            foreach (var attribute in _attributes)
                row[attribute] = null;

        if (!detections.Columns.Contains("track_id"))
            return detections;

        foreach (var tracklet in detections.Rows.Where(r => r.Has("track_id"))
                                                .GroupBy(r => r.Get<long>("track_id")))
        {
            foreach (var attribute in _attributes)
            {
                var value = AttributeVoting.SelectHighestVotedAttribute(
                    tracklet.Select(r => r.GetOrDefault<string>($"{attribute}_detection")),
                    tracklet.Select(r => r.GetOrDefault<double>($"{attribute}_confidence")));
                foreach (var row in tracklet)
                    row[attribute] = value;
            }
        }
        return detections;
    }
}
