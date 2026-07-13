using Microsoft.Extensions.Logging;
using OpenCvSharp;
using TrackLab.Datastruct;
using TrackLab.Pipeline;

namespace TrackLab.Engine;

/// <summary>
/// Port of tracklab.engine.offline.OfflineTrackingEngine: runs the whole pipeline
/// module-by-module over one video (each module sees the full video before the next starts).
/// </summary>
public sealed class OfflineTrackingEngine
{
    private readonly Pipeline.Pipeline _pipeline;
    private readonly TrackerState _trackerState;
    private readonly ILogger? _log;

    public OfflineTrackingEngine(Pipeline.Pipeline pipeline, TrackerState trackerState, ILogger? log = null)
    {
        _pipeline = pipeline;
        _trackerState = trackerState;
        _log = log;
    }

    public void TrackDataset()
    {
        foreach (var video in _trackerState.Set.VideoMetadatas.Rows)
        {
            var videoId = video.Get<long>("id");
            _log?.LogInformation("Tracking video {VideoId} ({Name})", videoId,
                video.GetOrDefault<string>("name"));
            var (detections, imagePreds) = VideoLoop(videoId);
            _trackerState.Update(videoId, detections, imagePreds);
        }
    }

    public (DataFrame detections, DataFrame imagePreds) VideoLoop(long videoId)
    {
        foreach (var model in _pipeline.Models)
            model.Reset();

        var (detections, imagePreds) = _trackerState.Load(videoId);

        foreach (var model in _pipeline.Models)
        {
            _log?.LogInformation("Running module {Module}", model.Name);
            switch (model)
            {
                case VideoLevelModule videoModule:
                    detections = videoModule.Process(detections, imagePreds);
                    break;
                case ImageLevelModule imageModule:
                    RunImageLevel(imageModule, detections, imagePreds);
                    break;
                case DetectionLevelModule detectionModule:
                    RunDetectionLevel(detectionModule, detections, imagePreds);
                    break;
                default:
                    throw new NotSupportedException($"Unknown module level: {model.Level}");
            }
            if (detections.IsEmpty && model.Level != "image")
                return (detections, imagePreds);
        }
        return (detections, imagePreds);
    }

    private void RunImageLevel(ImageLevelModule module, DataFrame detections, DataFrame imagePreds)
    {
        foreach (var metaBatch in imagePreds.Rows.Chunk(module.BatchSize))
        {
            var batch = new List<object>(metaBatch.Length);
            foreach (var metadata in metaBatch)
            {
                using var image = LoadImage(metadata);
                var dets = detections.Where(d => Equals(d["image_id"], metadata.Index));
                batch.Add(module.Preprocess(image, dets, metadata));
            }
            var output = module.Process(batch, detections, metaBatch);
            detections.Merge(output.Detections);
            if (output.ImageMetadatas is not null)
                imagePreds.Merge(output.ImageMetadatas);
        }
    }

    private void RunDetectionLevel(DetectionLevelModule module, DataFrame detections, DataFrame imagePreds)
    {
        // detections of one frame are contiguous, so a one-image cache avoids
        // re-reading the same file for every detection
        Mat? cachedImage = null;
        long cachedImageId = -1;
        try
        {
            var rows = detections.Rows.ToList();
            foreach (var detBatch in rows.Chunk(module.BatchSize))
            {
                var batch = new List<object>(detBatch.Length);
                var metas = new List<Series>(detBatch.Length);
                foreach (var detection in detBatch)
                {
                    var metadata = imagePreds[detection.Get<long>("image_id")];
                    metas.Add(metadata);
                    if (cachedImageId != metadata.Index)
                    {
                        cachedImage?.Dispose();
                        cachedImage = LoadImage(metadata);
                        cachedImageId = metadata.Index;
                    }
                    batch.Add(module.Preprocess(cachedImage!, detection, metadata));
                }
                detections.Merge(module.Process(batch, detections, metas));
            }
        }
        finally
        {
            cachedImage?.Dispose();
        }
    }

    private static Mat LoadImage(Series metadata)
    {
        var filePath = metadata.Get<string>("file_path");
        var image = Cv2.ImRead(filePath, ImreadModes.Color);
        if (image.Empty())
            throw new FileNotFoundException($"Could not load image {filePath}");
        return image;
    }
}
