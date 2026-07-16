namespace PitchWise.Tracking.Utils;

/// <summary>Confidence-weighted attribute voting over a tracklet.</summary>
public static class AttributeVoting
{
    /// <summary>Confidence-weighted majority vote over an attribute of a tracklet:
    /// each value's votes are summed by confidence, the highest total wins.
    /// Null values are ignored; returns null when nothing was voted.</summary>
    public static string? SelectHighestVotedAttribute(
        IEnumerable<string?> attributes, IEnumerable<double> confidences)
    {
        var votes = new Dictionary<string, double>();
        foreach (var (att, conf) in attributes.Zip(confidences))
        {
            if (string.IsNullOrEmpty(att)) continue;
            votes[att] = votes.GetValueOrDefault(att) + conf;
        }
        return votes.Count == 0 ? null : votes.MaxBy(kv => kv.Value).Key;
    }
}
