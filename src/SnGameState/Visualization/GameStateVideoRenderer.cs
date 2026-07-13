using OpenCvSharp;
using SnGameState.Calibration;
using TrackLab.Datastruct;

namespace SnGameState.Visualization;

/// <summary>
/// Counterpart of tracklab's VisualizationEngine for the gamestate config: renders the
/// tracking result of one video to an .mp4 with pitch lines, player ellipses and the
/// predictions radar (plus a ground-truth radar when GT detections are available).
/// </summary>
public sealed class GameStateVideoRenderer : IDisposable
{
    private readonly Mat _radarTemplate;

    public GameStateVideoRenderer(string? radarImagePath = null)
    {
        radarImagePath ??= Path.Combine(AppContext.BaseDirectory, "Assets", "Radar.png");
        _radarTemplate = File.Exists(radarImagePath)
            ? Cv2.ImRead(radarImagePath, ImreadModes.Color)
            : new Mat(new Size(125, 78), MatType.CV_8UC3, Scalar.White);
    }

    public void RenderVideo(TrackerState state, long videoId, string outputPath, double fps = 25)
    {
        var (detections, imagePreds) = state.Load(videoId);
        var gt = state.Set.DetectionsGt;

        using var writer = new VideoWriter();
        foreach (var metadata in imagePreds.Rows.OrderBy(r => r.GetOrDefault<long>("frame")))
        {
            using var image = Cv2.ImRead(metadata.Get<string>("file_path"), ImreadModes.Color);
            if (image.Empty()) continue;
            RenderFrame(image, metadata, detections, gt);

            if (!writer.IsOpened())
                writer.Open(outputPath, FourCC.MP4V, fps, image.Size());
            writer.Write(image);
        }
    }

    /// <summary>Draws one frame: pitch lines, per-detection ellipses and the radar(s).</summary>
    public void RenderFrame(Mat image, Series imageMetadata, DataFrame detections, DataFrame? detectionsGt = null)
    {
        if (imageMetadata["lines"] is IReadOnlyDictionary<string, List<Keypoint>> lines)
            GameStateVisualizer.DrawPitch(image, lines);

        var frameDetections = detections.Rows
            .Where(d => Equals(d["image_id"], imageMetadata.Index))
            .ToList();
        foreach (var detection in frameDetections)
            GameStateVisualizer.DrawPlayerEllipse(image, detection);

        GameStateVisualizer.DrawRadarView(image, frameDetections, _radarTemplate, group: "Predictions");

        if (detectionsGt is { IsEmpty: false })
        {
            var gtDetections = detectionsGt.Rows.Where(d => Equals(d["image_id"], imageMetadata.Index)).ToList();
            if (gtDetections.Count > 0)
                GameStateVisualizer.DrawRadarView(image, gtDetections, _radarTemplate, group: "Ground Truth");
        }
    }

    public void Dispose() => _radarTemplate.Dispose();
}
