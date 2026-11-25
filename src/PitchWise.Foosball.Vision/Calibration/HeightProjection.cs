using OpenCvSharp;
using PitchWise.Foosball.Vision.Tracking;

namespace PitchWise.Foosball.Vision.Calibration;

/// <summary>
/// Pixels ↔ table metres for things above the playfield. The homography alone puts them where their
/// line of sight meets the mat, beyond their real place; scaling towards the point below the camera
/// by the height ratio corrects that.
/// </summary>
public sealed class HeightProjection
{
    private readonly TableCalibration _calibration;
    private readonly double _cameraHeight;
    private readonly Point2d _belowCamera;

    public HeightProjection(TableCalibration calibration, TrackerOptions options)
    {
        _calibration = calibration;
        _cameraHeight = options.CameraHeight;
        _belowCamera = new Point2d(
            options.CameraX ?? calibration.Table.Length / 2,
            options.CameraY ?? calibration.Table.Width / 2);
    }

    public TableModel Table => _calibration.Table;

    /// <summary>Table position of a point seen at this pixel and lying <paramref name="height"/> above the mat.</summary>
    public Point2d ToTable(Point2d pixel, double height) =>
        Scale(_calibration.ToTable(pixel), (_cameraHeight - height) / _cameraHeight);

    /// <summary>Pixel where a point at this table position and height is seen.</summary>
    public Point2d ToImage(Point2d table, double height) =>
        _calibration.ToImage(Scale(table, _cameraHeight / (_cameraHeight - height)));

    private Point2d Scale(Point2d p, double factor) => _belowCamera + (p - _belowCamera) * factor;
}
