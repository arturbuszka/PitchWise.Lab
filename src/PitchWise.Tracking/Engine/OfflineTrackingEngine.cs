using Microsoft.Extensions.Logging;
using OpenCvSharp;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Pipeline;

namespace PitchWise.Tracking.Engine;

/// <summary>
/// Runs the whole pipeline
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

    /// <summary>
    /// Optional progress sink. Null (the default) leaves the engine exactly as it was.
    /// </summary>
    /// <remarks>
    /// Reported from the batch loops, so ticks arrive at frame granularity rather than once per
    /// module. The engine never marshals threads: an IProgress created on a UI thread does that
    /// for its owner, and throttling is the caller's business — batches are small and a run can
    /// report thousands of times.
    /// </remarks>
    public IProgress<TrackingProgress>? Progress { get; set; }

    /// <summary>
    /// Optional cooperative cancellation, checked once per batch so a cancelled run stops within
    /// one batch rather than at the next module boundary (which can be minutes away).
    /// </summary>
    public CancellationToken CancellationToken { get; set; } = CancellationToken.None;

    /// <summary>
    /// Optional module filter for runs seeded from a <see cref="StateCache"/> snapshot: modules
    /// it rejects are skipped (their cached columns stay), the others rerun on top of the cached
    /// state after their previous output columns are cleared. Null runs every module.
    /// </summary>
    public Func<Module, bool>? ShouldRun { get; set; }

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

        var moduleIndex = 0;
        var moduleCount = _pipeline.Models.Count;
        foreach (var model in _pipeline.Models)
        {
            CancellationToken.ThrowIfCancellationRequested();
            if (ShouldRun is not null)
            {
                if (!ShouldRun(model))
                {
                    _log?.LogInformation("Skipping module {Module} (cached)", model.Name);
                    moduleIndex++;
                    continue;
                }
                ClearOutputs(model, detections, imagePreds);
            }
            _log?.LogInformation("Running module {Module}", model.Name);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var reporter = new ModuleReporter(this, videoId, model.Name, moduleIndex++, moduleCount);
            switch (model)
            {
                case VideoLevelModule videoModule:
                    // atomic: nothing finer than "started"/"finished" exists to report
                    reporter.Report(0, 1);
                    detections = videoModule.Process(detections, imagePreds);
                    reporter.Report(1, 1);
                    break;
                case ImageLevelModule imageModule:
                    RunImageLevel(imageModule, detections, imagePreds, reporter);
                    break;
                case DetectionLevelModule detectionModule:
                    RunDetectionLevel(detectionModule, detections, imagePreds, reporter);
                    break;
                default:
                    throw new NotSupportedException($"Unknown module level: {model.Level}");
            }
            var perFrame = imagePreds.Count > 0 ? stopwatch.Elapsed.TotalSeconds / imagePreds.Count : 0;
            _log?.LogInformation("{Module}: {Total:F1}s ({PerFrame:F2}s/frame)",
                model.Name, stopwatch.Elapsed.TotalSeconds, perFrame);
            if (detections.IsEmpty && model.Level != "image")
                return (detections, imagePreds);
        }
        return (detections, imagePreds);
    }

    /// <summary>Drops a rerun module's previous outputs, so rows it no longer writes (e.g.
    /// detections a retuned tracker leaves unassigned) do not keep stale cached values.</summary>
    private static void ClearOutputs(Module module, DataFrame detections, DataFrame imagePreds)
    {
        foreach (var column in module.GetOutputColumns("detection"))
            foreach (var row in detections.Rows)
                row[column] = null;
        foreach (var column in module.GetOutputColumns("image"))
            foreach (var row in imagePreds.Rows)
                row[column] = null;
    }

    private void RunImageLevel(ImageLevelModule module, DataFrame detections, DataFrame imagePreds,
        ModuleReporter reporter)
    {
        long total = imagePreds.Count, done = 0;
        reporter.Report(done, total);
        foreach (var metaBatch in imagePreds.Rows.Chunk(module.BatchSize))
        {
            CancellationToken.ThrowIfCancellationRequested();
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
            reporter.Report(done += metaBatch.Length, total);
        }
    }

    private void RunDetectionLevel(DetectionLevelModule module, DataFrame detections, DataFrame imagePreds,
        ModuleReporter reporter)
    {
        // detections of one frame are contiguous, so a one-image cache avoids
        // re-reading the same file for every detection
        Mat? cachedImage = null;
        long cachedImageId = -1;
        try
        {
            var rows = detections.Rows.ToList();
            long total = rows.Count, done = 0;
            reporter.Report(done, total);
            foreach (var detBatch in rows.Chunk(module.BatchSize))
            {
                CancellationToken.ThrowIfCancellationRequested();
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
                reporter.Report(done += detBatch.Length, total);
            }
        }
        finally
        {
            cachedImage?.Dispose();
        }
    }

    /// <summary>Binds the per-module constants so the batch loops only supply the counts.</summary>
    private readonly struct ModuleReporter(
        OfflineTrackingEngine engine, long videoId, string moduleName, int moduleIndex, int moduleCount)
    {
        public void Report(long done, long total) => engine.Progress?.Report(
            new TrackingProgress(videoId, moduleName, moduleIndex, moduleCount, done, total));
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
