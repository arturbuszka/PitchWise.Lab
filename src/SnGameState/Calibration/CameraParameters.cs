namespace SnGameState.Calibration;

/// <summary>
/// Full camera parameters as produced by FramebyFrameCalib.get_cam_params — the
/// "parameters" image column of the pipeline (SoccerNet camera-format compatible).
/// </summary>
public sealed record CameraParameters
{
    public required double PanDegrees { get; init; }
    public required double TiltDegrees { get; init; }
    public required double RollDegrees { get; init; }
    public required double XFocalLength { get; init; }
    public required double YFocalLength { get; init; }
    public required double[] PrincipalPoint { get; init; }
    public required double[] PositionMeters { get; init; }
    public required double[,] RotationMatrix { get; init; }
    public double[] RadialDistortion { get; init; } = new double[6];
    public double[] TangentialDistortion { get; init; } = new double[2];
    public double[] ThinPrismDistortion { get; init; } = new double[4];
}
