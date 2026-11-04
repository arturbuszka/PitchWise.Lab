using System.Diagnostics;
using Microsoft.Extensions.Logging;
using PitchWise.App.Core.Workspace;
using PitchWise.Engine;
using PitchWise.GameState.Configuration;
using PitchWise.GameState.Dataset;
using PitchWise.GameState.Visualization;
using PitchWise.GameState.WorldState;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Engine;

namespace PitchWise.App.Core.Analysis;

/// <summary>
/// Runs analyses on a background thread, mirroring the CLI stats verb step for step.
/// </summary>
/// <remarks>
/// <para>
/// This is the async boundary of the whole application: below <see cref="RunCore"/> everything is
/// the ordinary synchronous library code the CLI calls, and no async keyword reaches
/// PitchWise.Tracking / GameState / Engine. A synchronous library invoked from a worker thread
/// composes strictly better than an async one — a future streaming backend can call the same
/// code from its own threading model without inheriting the desktop's choices.
/// </para>
/// <para>
/// One job at a time, deliberately: PipelineConfig.BuildPipeline sets the static
/// OnnxSessions.Provider/DeviceId, so two concurrent jobs with different configs would race over
/// which provider the sessions get. The GPU is the bottleneck anyway, so serialising costs
/// nothing. Do not add parallelism here without fixing that static first.
/// </para>
/// <para>
/// The CLI is the reference implementation: same input must yield the same MatchStatsReport and
/// the same world_state.jsonl. Any divergence is a bug here, not there.
/// </para>
/// </remarks>
public sealed class AnalysisJobRunner : IAnalysisJobRunner
{
    /// <summary>Cap on progress-report rate. Batches are small and a run reports thousands of
    /// times; ten updates a second is more than a human reads and cheap for the dispatcher.</summary>
    private static readonly TimeSpan ProgressInterval = TimeSpan.FromMilliseconds(100);

    private readonly IWorkspaceStore _store;
    private readonly ILoggerFactory _loggerFactory;

    public AnalysisJobRunner(IWorkspaceStore store, ILoggerFactory? loggerFactory = null)
    {
        _store = store;
        _loggerFactory = loggerFactory ?? Microsoft.Extensions.Logging.Abstractions.NullLoggerFactory.Instance;
    }

    public Task<AnalysisResult> RunAsync(WorkspacePaths paths, AnalysisRequest request,
        IProgress<AnalysisProgress> progress, CancellationToken ct)
        // LongRunning: minutes of solid CPU/GPU must not squat a thread-pool worker
        => Task.Factory.StartNew(() => RunCore(paths, request, progress, ct), ct,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);

    private AnalysisResult RunCore(WorkspacePaths paths, AnalysisRequest request,
        IProgress<AnalysisProgress> progress, CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();
        var manifest = _store.ReadManifest(paths.JobDirectory);
        using var log = new JobFileLogger(paths.Log, _loggerFactory.CreateLogger($"job:{paths.JobId}"));

        try
        {
            manifest = manifest with { Status = JobStatus.Running };
            _store.SaveManifest(paths, manifest);

            var set = LoadSource(request, paths, progress, stopwatch, log);
            ct.ThrowIfCancellationRequested();

            var video = set.VideoMetadatas.Rows.First();
            var videoId = video.Get<long>("id");
            var fps = video.Get<long>("frame_rate");
            var imWidth = (int)video.Get<long>("im_width");
            var imHeight = (int)video.Get<long>("im_height");
            log.LogInformation("{N} frames @ {Fps} fps ({W}x{H})",
                set.ImageMetadatas.Count, fps, imWidth, imHeight);

            var config = PipelineConfig.Load(request.ConfigPath);
            config.ModelsDir = ResolveModelsDir(request, config, log);
            var pipeline = config.BuildPipeline(imWidth, imHeight, log, fps);

            manifest = manifest with
            {
                FrameCount = set.ImageMetadatas.Count,
                Fps = fps,
                ImWidth = imWidth,
                ImHeight = imHeight,
                PipelineDescription = pipeline.ToString(),
            };
            _store.SaveManifest(paths, manifest);

            var costs = new ModuleCostModel(pipeline.Models.Select(m => m.Name).ToList());
            var state = new TrackerState(set);
            var engine = new OfflineTrackingEngine(pipeline, state, log)
            {
                CancellationToken = ct,
                Progress = new ThrottledProgress<TrackingProgress>(
                    tick => progress.Report(costs.ToAnalysisProgress(tick, stopwatch.Elapsed)),
                    ProgressInterval),
            };
            engine.TrackDataset();

            var (detections, imagePreds) = state.Load(videoId);

            // From here the steps mirror Program.Stats exactly, so the two agree by construction
            progress.Report(costs.ToStageProgress(AnalysisStage.Observing,
                "Collecting match statistics", 0.9, stopwatch.Elapsed));
            var observations = ObservationAdapter.ToObservations(detections, imagePreds, fps);

            using (var dump = WorldStateJsonl.TryCreate(paths.WorldStateJsonl,
                new WorldStateJsonl.Header(fps, request.FrameStride, PitchLength: 105, PitchWidth: 68,
                    NormalizedCoords: false)))
            {
                var report = new MatchStatsCollector().Run(observations, (obs, _) => dump?.Append(obs));
                _store.WriteStats(paths, report);
                log.LogInformation("{Frames} frames of statistics", report.FramesProcessed);
            }

            _store.WriteFrames(paths, detections, imagePreds);

            var artifacts = new ArtifactSet
            {
                Detections = WorkspacePaths.DetectionsFileName,
                ImagePreds = WorkspacePaths.ImagePredsFileName,
                Stats = WorkspacePaths.StatsFileName,
                WorldStateJsonl = WorkspacePaths.WorldStateJsonlFileName,
                Log = WorkspacePaths.LogFileName,
            };

            if (request.ExportVideo)
            {
                progress.Report(costs.ToStageProgress(AnalysisStage.Rendering,
                    "Rendering overlay video", 0.95, stopwatch.Elapsed));
                using var renderer = new GameStateVideoRenderer();
                renderer.RenderVideo(state, videoId, paths.Render, fps);
                artifacts = artifacts with { Render = WorkspacePaths.RenderFileName };
            }

            manifest = manifest with
            {
                Status = JobStatus.Completed,
                CompletedUtc = DateTimeOffset.UtcNow,
                Artifacts = artifacts,
            };
            _store.SaveManifest(paths, manifest);
            progress.Report(costs.ToStageProgress(AnalysisStage.Done, "Done", 1.0, stopwatch.Elapsed));
            log.LogInformation("Job {JobId} completed in {Elapsed:F1}s", paths.JobId,
                stopwatch.Elapsed.TotalSeconds);

            return new AnalysisResult(paths.JobId, paths.JobDirectory);
        }
        catch (OperationCanceledException)
        {
            _store.SaveManifest(paths, manifest with
            {
                Status = JobStatus.Cancelled,
                CompletedUtc = DateTimeOffset.UtcNow,
            });
            throw;
        }
        catch (Exception e)
        {
            // A failed job records why and stays in the queue; it must never take the app down.
            log.LogError(e, "Job {JobId} failed", paths.JobId);
            _store.SaveManifest(paths, manifest with
            {
                Status = JobStatus.Failed,
                CompletedUtc = DateTimeOffset.UtcNow,
                FailureMessage = e.Message,
            });
            throw;
        }
    }

    /// <summary>
    /// Works out where the ONNX models actually are.
    /// </summary>
    /// <remarks>
    /// The configs say <c>models_dir: pretrained_models</c>, a path relative to whatever the
    /// process's working directory happens to be. That is fine for the CLI, which is run from the
    /// repo root, but a GUI's working directory is its own bin folder, so the same config would
    /// look for the models under bin\Debug and fail. A relative directory is therefore resolved
    /// against the config's own location — the config and the models it names travel together —
    /// and against the working directory only as a last resort.
    /// </remarks>
    private static string ResolveModelsDir(AnalysisRequest request, PipelineConfig config, ILogger log)
    {
        // an explicit choice in the request wins, and AnalysisRequest has already made it absolute
        if (request.ModelsDir is { Length: > 0 } requested)
            return requested;

        if (Path.IsPathRooted(config.ModelsDir))
            return config.ModelsDir;

        if (request.ConfigPath is { Length: > 0 } configPath
            && Path.GetDirectoryName(Path.GetFullPath(configPath)) is { } configDir)
        {
            // configs/soccernet.yaml naming pretrained_models means <repo>/pretrained_models,
            // i.e. one level up from the config file itself
            foreach (var candidate in new[]
            {
                Path.GetFullPath(Path.Combine(configDir, config.ModelsDir)),
                Path.GetFullPath(Path.Combine(configDir, "..", config.ModelsDir)),
            })
                if (Directory.Exists(candidate))
                {
                    log.LogInformation("Models: {Dir} (resolved from the config's location)", candidate);
                    return candidate;
                }
        }

        var fallback = Path.GetFullPath(config.ModelsDir);
        log.LogInformation("Models: {Dir} (relative to the working directory)", fallback);
        return fallback;
    }

    private TrackingSet LoadSource(AnalysisRequest request, WorkspacePaths paths,
        IProgress<AnalysisProgress> progress, Stopwatch stopwatch, ILogger log)
    {
        progress.Report(new AnalysisProgress(AnalysisStage.Extracting, "Reading source", 0, 0, 0,
            stopwatch.Elapsed, null));
        log.LogInformation("Loading {Kind} from {Path}", request.SourceKind, request.SourcePath);

        return request.SourceKind switch
        {
            // extraction writes into the job's own folder, and skips frames already there
            SourceKind.VideoFile => ExternalVideo.Load(request.SourcePath, paths.FramesDir,
                request.FrameStride, request.MaxFrames, request.StartSeconds),

            // used in place: no copy, no re-encode, so the job folder stays empty of frames
            SourceKind.FrameSequence => FrameSequenceSet.Load(request.SourcePath, request.SourceFps,
                request.FrameStride, request.MaxFrames),

            SourceKind.SoccerNetVideo => TrimSet(SoccerNetGameState.LoadVideo(request.SourcePath),
                request.MaxFrames),

            _ => throw new NotSupportedException($"unknown source kind {request.SourceKind}"),
        };
    }

    /// <summary>
    /// Caps a loaded set to its first n frames. SoccerNet sets are loaded whole from their
    /// labels file, so unlike the other two loaders there is no maxFrames to pass down.
    /// </summary>
    private static TrackingSet TrimSet(TrackingSet set, int maxFrames)
    {
        if (maxFrames <= 0 || set.ImageMetadatas.Count <= maxFrames) return set;

        var keptImages = new DataFrame();
        var keptIds = new HashSet<long>();
        foreach (var image in set.ImageMetadatas.Rows
            .OrderBy(r => r.GetOrDefault<long>("frame")).Take(maxFrames))
        {
            keptImages.Upsert(image);
            keptIds.Add(image.Index);
        }
        return new TrackingSet
        {
            VideoMetadatas = set.VideoMetadatas,
            ImageMetadatas = keptImages,
            DetectionsGt = set.DetectionsGt.Where(d => keptIds.Contains(d.Get<long>("image_id"))),
        };
    }

    /// <summary>
    /// Passes at most one report per interval through to the inner handler.
    /// </summary>
    /// <remarks>
    /// Dropping the tail of a burst is safe here: the tracking loop is always followed by an
    /// explicit stage report, so the bar never comes to rest on a throttled-away value.
    /// </remarks>
    private sealed class ThrottledProgress<T>(Action<T> handler, TimeSpan interval) : IProgress<T>
    {
        private readonly Stopwatch _sinceLast = Stopwatch.StartNew();

        public void Report(T value)
        {
            if (_sinceLast.Elapsed < interval) return;
            _sinceLast.Restart();
            handler(value);
        }
    }
}
