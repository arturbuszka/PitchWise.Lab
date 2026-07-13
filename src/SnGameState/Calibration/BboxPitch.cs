namespace SnGameState.Calibration;

/// <summary>
/// Pitch-coordinate projection of a detection bbox, mirroring the "bbox_pitch" dict produced
/// by sn_gamestate.calibration (origin at pitch center, x towards the right goal, in meters).
/// </summary>
public sealed record BboxPitch(
    double XBottomLeft, double YBottomLeft,
    double XBottomRight, double YBottomRight,
    double XBottomMiddle, double YBottomMiddle);
