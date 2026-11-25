using System.Text.Json;
using System.Text.Json.Serialization;

namespace PitchWise.Foosball.Vision.Tracking;

/// <summary>Ball and rod tracking settings in metres and seconds, read from <c>configs/foosball/tracker.json</c>.</summary>
/// <param name="BallHeight">Height of the ball centre above the playfield.</param>
/// <param name="CameraHeight">Height of the lens above the playfield.</param>
/// <param name="OutsideMargin">How far outside the playfield a ball is still believed.</param>
/// <param name="CameraX">Table x straight below the camera; null means the table centre.</param>
/// <param name="CameraY">Table y straight below the camera; null means the table centre.</param>
/// <param name="TrailSeconds">Length of the path drawn on review videos.</param>
/// <param name="RodHeight">Height of the rod axis above the playfield.</param>
/// <param name="RodAssignDistance">A figure further than this from every rod, along x, is ignored.</param>
/// <param name="RodFitTolerance">How far a figure may sit from where its rod says it should be.</param>
/// <param name="RodMaxSpeed">Faster slides are taken for a mistake, metres per second.</param>
/// <param name="RodMaxHoldSeconds">How long a rod with no visible figures keeps its last position.</param>
/// <param name="RodSmoothingSeconds">Time constant of the offset smoothing.</param>
public sealed record TrackerOptions(
    [property: JsonRequired] double BallHeight,
    [property: JsonRequired] double CameraHeight,
    [property: JsonRequired] double OutsideMargin,
    double? CameraX = null,
    double? CameraY = null,
    double TrailSeconds = 0.5,
    double RodHeight = 0.050,
    double RodAssignDistance = 0.040,
    double RodFitTolerance = 0.012,
    double RodMaxSpeed = 5.0,
    double RodMaxHoldSeconds = 0.5,
    double RodSmoothingSeconds = 0.005)
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static TrackerOptions Load(string path)
    {
        var options = JsonSerializer.Deserialize<TrackerOptions>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"Empty tracker config {path}");
        if (options.BallHeight < 0 || options.CameraHeight <= options.BallHeight)
            throw new InvalidDataException(
                $"Camera height {options.CameraHeight} must be above ball height {options.BallHeight} in {path}");
        return options;
    }
}
