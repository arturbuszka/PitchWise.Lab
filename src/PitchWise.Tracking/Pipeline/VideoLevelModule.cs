using PitchWise.Tracking.Datastruct;

namespace PitchWise.Tracking.Pipeline;

/// <summary>
/// A module that
/// operates on a whole video at once (e.g. tracklet-level voting or team clustering).
/// </summary>
public abstract class VideoLevelModule : Module
{
    public override string Level => "video";

    /// <summary>Processes all detections of one video; returns updated detections.</summary>
    public abstract DataFrame Process(DataFrame detections, DataFrame imageMetadatas);
}
