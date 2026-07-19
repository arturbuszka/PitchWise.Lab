using System.Text.Json;
using Microsoft.Extensions.Logging;
using PitchWise.Engine;
using PitchWise.GameState.WorldState;
using PitchWise.GameState.Calibration;
using PitchWise.GameState.Configuration;
using PitchWise.GameState.Dataset;
using PitchWise.GameState.Evaluation;
using PitchWise.GameState.Detection;
using PitchWise.GameState.Jersey;
using PitchWise.GameState.Reid;
using PitchWise.GameState.Team;
using PitchWise.GameState.Track;
using PitchWise.GameState.Visualization;
using PitchWise.Tracking.Datastruct;
using PitchWise.Tracking.Engine;
using PitchWise.Tracking.Pipeline;
using PitchWise.Tracking.Wrappers;

using var loggerFactory = LoggerFactory.Create(b => b.AddSimpleConsole());
var log = loggerFactory.CreateLogger("pitchwise");

// -c <config.yaml> anywhere in the args; defaults to configs/soccernet.yaml when present
var argList = args.ToList();
string? configPath = null;
var configIdx = argList.IndexOf("-c");
if (configIdx >= 0 && configIdx + 1 < argList.Count)
{
    configPath = argList[configIdx + 1];
    argList.RemoveRange(configIdx, 2);
}
configPath ??= File.Exists("configs/soccernet.yaml") ? "configs/soccernet.yaml" : null;
var config = PipelineConfig.Load(configPath);
if (configPath is not null)
    log.LogInformation("Config: {Config}", configPath);
args = argList.ToArray();

return args switch
{
    ["render", var dataset, var split, var video, var output] => Render(dataset, split, video, output),
    ["track", var video, var output, .. var rest] => Track(video, output,
        rest.Length > 0 ? rest[0] : null,
        rest.Length > 1 ? int.Parse(rest[1]) : 0,
        rest.Length > 2 ? double.Parse(rest[2]) : 0),
    ["stats", var video, var output, .. var rest] => Stats(video, output,
        rest.Length > 0 ? rest[0] : null,
        rest.Length > 1 ? int.Parse(rest[1]) : 0,
        rest.Length > 2 ? double.Parse(rest[2]) : 0),
    ["eval", var dataset, var split, var video, .. var rest] => Eval(dataset, split, video,
        rest.Length > 0 ? rest[0] : null,
        rest.Length > 1 ? int.Parse(rest[1]) : 0),
    _ => Usage(),
};

int Usage()
{
    Console.WriteLine("""
        usage:
          render <dataset_path> <split> <video_name> <out.mp4>   render dataset ground truth
          track <video.mp4> <out.mp4> [models_dir] [max_frames] [start_seconds]  run the pipeline on any video
          stats <video.mp4> <out.json> [models_dir] [max_frames] [start_seconds] run the pipeline + collect match statistics
          eval <dataset_path> <split> <video_name> [models_dir] [max_frames]     run pipeline + GS-HOTA vs ground truth
        """);
    return 1;
}

int Render(string datasetPath, string split, string videoName, string output)
{
    var videoFolder = Path.Combine(datasetPath, split, videoName);
    if (!Directory.Exists(videoFolder))
    {
        Console.WriteLine($"video folder not found: {videoFolder}");
        return 1;
    }
    log.LogInformation("Loading {Folder}", videoFolder);
    var set = SoccerNetGameState.LoadVideo(videoFolder);
    var videoId = set.VideoMetadatas.Rows.First().Get<long>("id");

    var state = new TrackerState(set);
    var (detections, imagePreds) = state.Load(videoId);
    foreach (var gt in set.DetectionsGt.Rows.Where(d => Equals(d["video_id"], videoId)))
        detections.Upsert(gt);
    state.Update(videoId, detections, imagePreds);

    log.LogInformation("Rendering {N} frames of {Video} to {Output}", imagePreds.Count, videoName, output);
    using var renderer = new GameStateVideoRenderer();
    renderer.RenderVideo(state, videoId, output);
    log.LogInformation("Done");
    return 0;
}

Pipeline BuildPipeline(string? modelsDir, int imWidth, int imHeight)
{
    if (modelsDir is not null)
        config.ModelsDir = modelsDir;
    return config.BuildPipeline(imWidth, imHeight, log);
}

int Track(string videoPath, string output, string? modelsDir, int maxFrames, double startSeconds = 0)
{

    var framesDir = Path.Combine(
        Path.GetDirectoryName(Path.GetFullPath(output))!,
        $"frames-{Path.GetFileNameWithoutExtension(videoPath)}");
    log.LogInformation("Extracting frames of {Video} to {Dir}", videoPath, framesDir);
    var set = ExternalVideo.Load(videoPath, framesDir, maxFrames: maxFrames, startSeconds: startSeconds);
    var videoId = set.VideoMetadatas.Rows.First().Get<long>("id");
    log.LogInformation("{N} frames", set.ImageMetadatas.Count);

    var imWidth = (int)set.VideoMetadatas.Rows.First().Get<long>("im_width");
    var imHeight = (int)set.VideoMetadatas.Rows.First().Get<long>("im_height");
    var pipeline = BuildPipeline(modelsDir, imWidth, imHeight);

    var state = new TrackerState(set);
    var engine = new OfflineTrackingEngine(pipeline, state, log);
    engine.TrackDataset();

    log.LogInformation("Rendering to {Output}", output);
    using var renderer = new GameStateVideoRenderer();
    renderer.RenderVideo(state, videoId, output);
    log.LogInformation("Done");
    return 0;
}

int Stats(string videoPath, string output, string? modelsDir, int maxFrames, double startSeconds = 0)
{
    var framesDir = Path.Combine(
        Path.GetDirectoryName(Path.GetFullPath(output))!,
        $"frames-{Path.GetFileNameWithoutExtension(videoPath)}");
    log.LogInformation("Extracting frames of {Video} to {Dir}", videoPath, framesDir);
    var set = ExternalVideo.Load(videoPath, framesDir, maxFrames: maxFrames, startSeconds: startSeconds);
    var video = set.VideoMetadatas.Rows.First();
    var videoId = video.Get<long>("id");
    var fps = video.Get<long>("frame_rate");
    log.LogInformation("{N} frames @ {Fps} fps", set.ImageMetadatas.Count, fps);

    var imWidth = (int)video.Get<long>("im_width");
    var imHeight = (int)video.Get<long>("im_height");
    var pipeline = BuildPipeline(modelsDir, imWidth, imHeight);

    var state = new TrackerState(set);
    new OfflineTrackingEngine(pipeline, state, log).TrackDataset();
    var (detections, imagePreds) = state.Load(videoId);

    log.LogInformation("Collecting match statistics");
    var observations = ObservationAdapter.ToObservations(detections, imagePreds, fps);

    // Dump the observation stream alongside the report, so EngineConfig thresholds can be
    // re-tuned via WorldStateReplay without decoding the video again.
    var jsonlPath = Path.ChangeExtension(output, ".world_state.jsonl");
    using var dump = WorldStateJsonl.TryCreate(jsonlPath,
        new WorldStateJsonl.Header(fps, FrameStride: 1, PitchLength: 105, PitchWidth: 68, NormalizedCoords: false));

    var collector = new MatchStatsCollector();
    var report = collector.Run(observations, (obs, _) => dump?.Append(obs));

    var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(output, json);

    log.LogInformation("Wrote {Output} ({Frames} frames)", output, report.FramesProcessed);
    Console.WriteLine();
    Console.WriteLine($"Possession — A: {report.TeamA.PossessionPct}%  B: {report.TeamB.PossessionPct}%  " +
                      $"(controlled {report.ControlledSeconds:F1}s, loose {report.LooseSeconds:F1}s)");
    Console.WriteLine($"Passes     — A: {report.TeamA.Passes} ({report.TeamA.PassAccuracyPct}% acc, {report.TeamA.Turnovers} lost)  " +
                      $"B: {report.TeamB.Passes} ({report.TeamB.PassAccuracyPct}% acc, {report.TeamB.Turnovers} lost)");
    Console.WriteLine($"Players    — {report.Players.Count} tracked; top distance: " +
                      string.Join(", ", report.Players.Take(3).Select(p => $"#{p.TrackId} {p.DistanceMeters:F0}m")));
    return 0;
}

int Eval(string datasetPath, string split, string videoName, string? modelsDir, int maxFrames)
{
    var videoFolder = Path.Combine(datasetPath, split, videoName);
    if (!Directory.Exists(videoFolder))
    {
        Console.WriteLine($"video folder not found: {videoFolder}");
        return 1;
    }
    log.LogInformation("Loading {Folder}", videoFolder);
    var set = SoccerNetGameState.LoadVideo(videoFolder);
    if (maxFrames > 0)
        set = TrimSet(set, maxFrames);
    var videoId = set.VideoMetadatas.Rows.First().Get<long>("id");
    var imWidth = (int)set.VideoMetadatas.Rows.First().Get<long>("im_width");
    var imHeight = (int)set.VideoMetadatas.Rows.First().Get<long>("im_height");

    var pipeline = BuildPipeline(modelsDir, imWidth, imHeight);
    var state = new TrackerState(set);
    new OfflineTrackingEngine(pipeline, state, log).TrackDataset();
    var (predictions, _) = state.Load(videoId);

    // frames in GT order; predictions must have an identity and a pitch position
    // (mirrors the dropna in the Python save_for_eval)
    var imageIds = set.ImagesOfVideo(videoId).Select(m => m.Index).ToList();
    var gtFrames = new List<IReadOnlyList<GsDetection>>();
    var predFrames = new List<IReadOnlyList<GsDetection>>();
    foreach (var imageId in imageIds)
    {
        gtFrames.Add(set.DetectionsGt.Rows
            .Where(d => Equals(d["image_id"], imageId) && d["bbox_pitch"] is BboxPitch
                        && d.GetOrDefault<string>("role") != "ball")
            .Select(d => ToGsDetection(d))
            .ToList());
        predFrames.Add(predictions.Rows
            .Where(d => Equals(d["image_id"], imageId) && d.Has("track_id") && d["bbox_pitch"] is BboxPitch)
            .Select(d => ToGsDetection(d))
            .ToList());
    }

    var full = GsHota.Evaluate(gtFrames, predFrames);
    var noJersey = GsHota.Evaluate(gtFrames, predFrames, new GsHotaOptions { UseJerseyNumbers = false });
    var positionOnly = GsHota.Evaluate(gtFrames, predFrames,
        new GsHotaOptions { UseJerseyNumbers = false, UseTeams = false, UseRoles = false });

    Console.WriteLine();
    Console.WriteLine($"GS-HOTA ({videoName}, {imageIds.Count} frames, {gtFrames.Sum(f => f.Count)} GT / {predFrames.Sum(f => f.Count)} pred detections)");
    Console.WriteLine($"  full (role+team+jersey): HOTA={full.Hota:P1}  DetA={full.DetA:P1}  AssA={full.AssA:P1}  LocA={full.LocA:P1}");
    Console.WriteLine($"  without jersey numbers:  HOTA={noJersey.Hota:P1}  DetA={noJersey.DetA:P1}  AssA={noJersey.AssA:P1}");
    Console.WriteLine($"  position only:           HOTA={positionOnly.Hota:P1}  DetA={positionOnly.DetA:P1}  AssA={positionOnly.AssA:P1}");
    return 0;
}

static GsDetection ToGsDetection(Series d)
{
    var pitch = d.Get<BboxPitch>("bbox_pitch");
    return new GsDetection(
        d.GetOrDefault<long>("track_id"),
        pitch.XBottomMiddle, pitch.YBottomMiddle,
        d.GetOrDefault<string>("role"),
        d.GetOrDefault<string>("team"),
        d.GetOrDefault<string>("jersey_number"));
}

static PitchWise.Tracking.Datastruct.TrackingSet TrimSet(PitchWise.Tracking.Datastruct.TrackingSet set, int maxFrames)
{
    var keptImages = new DataFrame();
    var keptIds = new HashSet<long>();
    foreach (var image in set.ImageMetadatas.Rows.OrderBy(r => r.GetOrDefault<long>("frame")).Take(maxFrames))
    {
        keptImages.Upsert(image);
        keptIds.Add(image.Index);
    }
    return new PitchWise.Tracking.Datastruct.TrackingSet
    {
        VideoMetadatas = set.VideoMetadatas,
        ImageMetadatas = keptImages,
        DetectionsGt = set.DetectionsGt.Where(d => keptIds.Contains(d.Get<long>("image_id"))),
    };
}
