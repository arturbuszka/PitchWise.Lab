namespace PitchWise.GameState.Calibration;

/// <summary>
/// Pitch-coordinate projection of a detection bbox, mirroring the "bbox_pitch" dict produced
/// (origin at pitch center, x towards the right goal, in meters).
/// </summary>
public sealed record BboxPitch(
    double XBottomLeft, double YBottomLeft,
    double XBottomRight, double YBottomRight,
    double XBottomMiddle, double YBottomMiddle);
