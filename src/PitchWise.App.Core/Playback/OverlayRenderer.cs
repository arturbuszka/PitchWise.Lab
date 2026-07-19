using OpenCvSharp;
using PitchWise.GameState.Calibration;
using PitchWise.GameState.Visualization;
using PitchWise.Tracking.Datastruct;

namespace PitchWise.App.Core.Playback;

/// <summary>What to draw over the frame.</summary>
/// <param name="Pitch">Calibrated pitch lines.</param>
/// <param name="Ellipses">Per-player ellipse and label.</param>
/// <param name="Radar">Bird's-eye radar, drawn into the frame's corner.</param>
/// <param name="GroundTruthRadar">Second radar from the dataset's ground truth, when there is one.</param>
/// <param name="TrackIds">Track id on the player label.</param>
/// <param name="JerseyNumbers">Jersey number on the player label.</param>
/// <param name="Roles">Role marker on the player label.</param>
public readonly record struct OverlayOptions(
    bool Pitch = true,
    bool Ellipses = true,
    bool Radar = true,
    bool GroundTruthRadar = false,
    bool TrackIds = true,
    bool JerseyNumbers = true,
    bool Roles = true)
{
    public static OverlayOptions Default => new();
}

/// <summary>
/// Draws the analysis overlay onto a decoded frame.
/// </summary>
/// <remarks>
/// <para>
/// Makes the same GameStateVisualizer calls, in the same order, as
/// GameStateVideoRenderer.RenderFrame — so what is scrubbed on screen and what an exported mp4
/// contains agree. They are called individually rather than through RenderFrame only to honour
/// the toggles, which needs no library change because all three draw methods are public.
/// </para>
/// <para>
/// Detections are indexed by image_id once, up front. RenderFrame rescans the whole detection
/// frame per call, which is fine for a one-pass export but not for a scrub-and-replay UI: a full
/// match's detections rescanned 25 times a second would stall playback.
/// </para>
/// </remarks>
public sealed class OverlayRenderer : IDisposable
{
    private readonly Dictionary<long, List<Series>> _byImage;
    private readonly Dictionary<long, List<Series>> _groundTruthByImage;
    private readonly Mat _radarTemplate = LoadRadarTemplate();

    public OverlayRenderer(DataFrame detections, DataFrame? groundTruth = null)
    {
        _byImage = IndexByImage(detections);
        _groundTruthByImage = groundTruth is null ? [] : IndexByImage(groundTruth);
    }

    /// <summary>True when the job carries ground truth, so the UI can offer that toggle.</summary>
    public bool HasGroundTruth => _groundTruthByImage.Count > 0;

    /// <summary>Detections of one frame, in the order the modules produced them.</summary>
    public IReadOnlyList<Series> DetectionsOf(long imageId)
        => _byImage.TryGetValue(imageId, out var rows) ? rows : [];

    /// <summary>Draws onto <paramref name="frame"/> in place.</summary>
    public void Draw(Mat frame, Series imageMetadata, OverlayOptions options)
    {
        if (options.Pitch && imageMetadata["lines"] is IReadOnlyDictionary<string, List<Keypoint>> lines)
            GameStateVisualizer.DrawPitch(frame, lines);

        var detections = DetectionsOf(imageMetadata.Index);

        if (options.Ellipses)
            foreach (var detection in detections)
                GameStateVisualizer.DrawPlayerEllipse(frame, detection,
                    displayTrackId: options.TrackIds,
                    displayJersey: options.JerseyNumbers,
                    displayRole: options.Roles);

        if (options.Radar && detections.Count > 0)
            GameStateVisualizer.DrawRadarView(frame, detections, _radarTemplate, group: "Predictions");

        if (options.GroundTruthRadar
            && _groundTruthByImage.TryGetValue(imageMetadata.Index, out var gt) && gt.Count > 0)
            GameStateVisualizer.DrawRadarView(frame, gt, _radarTemplate, group: "Ground Truth");
    }

    /// <summary>
    /// Loaded the same way GameStateVideoRenderer loads it, including the silent white fallback:
    /// an ugly radar beats a crash. If the radar looks blank, Assets\Radar.png did not reach the
    /// output directory.
    /// </summary>
    private static Mat LoadRadarTemplate()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "Radar.png");
        return File.Exists(path)
            ? Cv2.ImRead(path, ImreadModes.Color)
            : new Mat(new Size(125, 78), MatType.CV_8UC3, Scalar.White);
    }

    private static Dictionary<long, List<Series>> IndexByImage(DataFrame frame)
    {
        var index = new Dictionary<long, List<Series>>();
        foreach (var row in frame.Rows)
        {
            if (row["image_id"] is not long imageId) continue;
            if (!index.TryGetValue(imageId, out var rows))
                index[imageId] = rows = [];
            rows.Add(row);
        }
        return index;
    }

    public void Dispose() => _radarTemplate.Dispose();
}
