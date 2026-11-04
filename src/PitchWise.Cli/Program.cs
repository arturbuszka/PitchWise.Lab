using System.Text.Json;
using Microsoft.Extensions.Logging;
using PitchWise.Engine;
using PitchWise.Engine.Rules;
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
using PitchWise.Tracking.Eval;
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
// --dump <file.json>: eval writes its metrics and the evaluated detections there (for eval-combine)
string? dumpPath = null;
var dumpIdx = argList.IndexOf("--dump");
if (dumpIdx >= 0 && dumpIdx + 1 < argList.Count)
{
    dumpPath = argList[dumpIdx + 1];
    argList.RemoveRange(dumpIdx, 2);
}
// state cache: --cache <dir> saves each video's full pipeline state after eval; --from-cache <dir>
// seeds eval from it and reruns only the --rerun modules (comma-separated) plus every module that
// consumes their outputs — e.g. retune the tracker without detection, calibration and OCR
string? TakeOption(string name)
{
    var idx = argList.IndexOf(name);
    if (idx < 0 || idx + 1 >= argList.Count)
        return null;
    var value = argList[idx + 1];
    argList.RemoveRange(idx, 2);
    return value;
}
var cacheDir = TakeOption("--cache");
var fromCacheDir = TakeOption("--from-cache");
var rerun = TakeOption("--rerun")?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            ?? Array.Empty<string>();
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
    ["analyze", var video, var outDir, .. var rest] => Analyze(video, outDir,
        rest.Length > 0 ? rest[0] : null,
        rest.Length > 1 ? int.Parse(rest[1]) : 0),
    ["eval", var dataset, var split, var video, .. var rest] => Eval(dataset, split, video,
        rest.Length > 0 ? rest[0] : null,
        rest.Length > 1 ? int.Parse(rest[1]) : 0),
    ["eval-combine", var output, .. var inputs] when inputs.Length > 0 => EvalCombine(output, inputs),
    _ => Usage(),
};

int Usage()
{
    Console.WriteLine("""
        usage:
          render <dataset_path> <split> <video_name> <out.mp4>   render dataset ground truth
          track <video.mp4> <out.mp4> [models_dir] [max_frames] [start_seconds]  run the pipeline on any video
          stats <video.mp4> <out.json> [models_dir] [max_frames] [start_seconds] run the pipeline + collect match statistics
          analyze <video.mp4> <out_dir> [models_dir] [max_frames]                for other programs: writes result.json,
                                                                        world_state.jsonl and render.mp4; prints "PROGRESS 0.42"
          eval <dataset_path> <split> <video_name> [models_dir] [max_frames]     run pipeline + GS-HOTA vs ground truth
                                                                        (--dump <file.json> saves metrics + detections,
                                                                         --cache <dir> saves the pipeline state,
                                                                         --from-cache <dir> --rerun <Module,..> reuses it)
          eval-combine <out.json> <dump.json>...   GS-HOTA over several eval dumps, combined like TrackEval
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

Pipeline BuildPipeline(string? modelsDir, Series video)
{
    if (modelsDir is not null)
        config.ModelsDir = modelsDir;
    return config.BuildPipeline((int)video.Get<long>("im_width"), (int)video.Get<long>("im_height"), log,
        video.GetOrDefault<long>("frame_rate"));
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

    var pipeline = BuildPipeline(modelsDir, set.VideoMetadatas.Rows.First());

    var state = new TrackerState(set);
    var engine = new OfflineTrackingEngine(pipeline, state, log);
    engine.TrackDataset();

    log.LogInformation("Rendering to {Output}", output);
    using var renderer = new GameStateVideoRenderer();
    renderer.RenderVideo(state, videoId, output, set.VideoMetadatas.Rows.First().GetOrDefault<long>("frame_rate"));
    log.LogInformation("Done");
    return 0;
}

// The entry point for programs that run this CLI as a separate process: everything goes in one
// folder, in the documented formats, and a failure is a non-zero exit code.
int Analyze(string videoPath, string outDir, string? modelsDir, int maxFrames)
{
    Directory.CreateDirectory(outDir);
    var framesDir = Path.Combine(outDir, "frames");
    var jsonlPath = Path.Combine(outDir, "world_state.jsonl");
    try
    {
        var set = ExternalVideo.Load(videoPath, framesDir, maxFrames: maxFrames);
        var video = set.VideoMetadatas.Rows.First();
        var videoId = video.Get<long>("id");
        var fps = video.Get<long>("frame_rate");

        var pipeline = BuildPipeline(modelsDir, video);
        var state = new TrackerState(set);
        var lastPercent = -1;
        new OfflineTrackingEngine(pipeline, state, log)
        {
            Progress = new InlineProgress(tick =>
            {
                var percent = (int)(100 * (tick.ModuleIndex + tick.ModuleFraction) / tick.ModuleCount);
                if (percent == lastPercent) return;
                lastPercent = percent;
                Console.WriteLine(FormattableString.Invariant($"PROGRESS {percent / 100.0:F2}"));
            }),
        }.TrackDataset();

        var (detections, imagePreds) = state.Load(videoId);
        var observations = ObservationAdapter.ToObservations(detections, imagePreds, fps);
        MatchStatsReport report;
        using (var dump = WorldStateJsonl.TryCreate(jsonlPath,
            new WorldStateJsonl.Header(fps, FrameStride: 1, PitchLength: 105, PitchWidth: 68, NormalizedCoords: false)))
        {
            report = new MatchStatsCollector().Run(observations, (obs, _) => dump?.Append(obs));
        }

        var cfg = new EngineConfig();
        var events = WorldStateReplay.Run(jsonlPath, [new PassRule(cfg), new PossessionRule()], cfg).Events;

        using (var renderer = new GameStateVideoRenderer())
            renderer.RenderVideo(state, videoId, Path.Combine(outDir, "render.mp4"), fps);

        // written last, so its presence means the whole analysis finished
        File.WriteAllText(Path.Combine(outDir, "result.json"),
            JsonSerializer.Serialize(AnalysisResult.Create(fps, report, events), AnalysisResult.JsonOptions));
        log.LogInformation("Done: {Frames} frames, {Events} events", report.FramesProcessed, events.Count);
        return 0;
    }
    finally
    {
        if (Directory.Exists(framesDir)) Directory.Delete(framesDir, recursive: true);
    }
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

    var pipeline = BuildPipeline(modelsDir, video);

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

    var pipeline = BuildPipeline(modelsDir, set.VideoMetadatas.Rows.First());
    var state = new TrackerState(set);
    var engine = new OfflineTrackingEngine(pipeline, state, log);
    var fingerprints = config.ModuleFingerprints();
    if (fromCacheDir is not null)
    {
        var snapshot = StateCache.Load(StatePath(fromCacheDir, videoName));
        state.Update(videoId, snapshot.Detections, snapshot.ImagePreds);
        var toRun = ModulesToRerun(pipeline, rerun);
        // a cached module whose config or model changed would silently reuse stale outputs
        var stale = pipeline.Models
            .Where(m => !toRun.Contains(m.Name) && fingerprints.TryGetValue(m.Name, out var fp)
                        && snapshot.Meta.GetValueOrDefault(m.Name) != fp)
            .Select(m => m.Name).ToList();
        if (stale.Count > 0)
            throw new InvalidOperationException(
                $"cache {fromCacheDir} is stale for {string.Join(", ", stale)} (config or model changed) — add them to --rerun");
        log.LogInformation("From cache {Dir}: rerunning {Modules}", fromCacheDir, string.Join(" -> ", toRun));
        engine.ShouldRun = m => toRun.Contains(m.Name);
    }
    engine.TrackDataset();
    var (predictions, predImages) = state.Load(videoId);
    if (cacheDir is not null)
    {
        StateCache.Save(StatePath(cacheDir, videoName), predictions, predImages, fingerprints);
        log.LogInformation("State cached to {Path}", StatePath(cacheDir, videoName));
    }

    // frames in GT order, only those labelled for evaluation (has_labeled_pitch/camera/person, as
    // sn-trackeval); predictions must have an identity and a pitch position (mirrors the dropna in
    // the Python save_for_eval); the ball is ignored on both sides (IGNORE_BALL)
    var imageIds = set.ImagesOfVideo(videoId)
        .Where(m => new[] { "has_labeled_pitch", "has_labeled_camera", "has_labeled_person" }
            .All(flag => m.GetOrDefault(flag, true) == true))
        .Select(m => m.Index).ToList();
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
            .Where(d => Equals(d["image_id"], imageId) && d.Has("track_id") && d["bbox_pitch"] is BboxPitch
                        && d.GetOrDefault<string>("role") != "ball")
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

    if (dumpPath is not null)
    {
        var dump = new EvalDump(videoName, imageIds.Count, full, noJersey, positionOnly, imageIds, gtFrames, predFrames);
        File.WriteAllText(dumpPath, JsonSerializer.Serialize(dump));
        log.LogInformation("Eval dump written to {Path}", dumpPath);
    }
    return 0;
}

static string StatePath(string dir, string videoName) => Path.Combine(dir, $"{videoName}.state.json.gz");

// The requested modules plus, in pipeline order, every module reading a column they (or an earlier
// rerun module) write.
static List<string> ModulesToRerun(Pipeline pipeline, IReadOnlyCollection<string> requested)
{
    var unknown = requested.Except(pipeline.Models.Select(m => m.Name)).ToList();
    if (unknown.Count > 0)
        throw new ArgumentException($"--rerun: unknown module(s) {string.Join(", ", unknown)}; pipeline has "
                                    + string.Join(", ", pipeline.Models.Select(m => m.Name)));
    // A module that rewrites a column it reads (TrackletMerging: track_id) must see the column as its
    // producer wrote it, not its own cached output: rerunning it brings back the earlier writers.
    var modules = pipeline.Models.ToList();
    var wanted = requested.ToHashSet();
    for (var i = modules.Count - 1; i >= 0; i--)
    {
        if (!wanted.Contains(modules[i].Name))
            continue;
        var inPlace = modules[i].GetInputColumns("detection").Intersect(modules[i].GetOutputColumns("detection")).ToList();
        foreach (var producer in modules.Take(i).Where(m => m.GetOutputColumns("detection").Intersect(inPlace).Any()))
            wanted.Add(producer.Name);
    }
    var dirty = new HashSet<string>();
    var result = new List<string>();
    foreach (var module in pipeline.Models)
    {
        var inputs = module.GetInputColumns("detection").Select(c => "detection:" + c)
            .Concat(module.GetInputColumns("image").Select(c => "image:" + c));
        if (!wanted.Contains(module.Name) && !inputs.Any(dirty.Contains))
            continue;
        result.Add(module.Name);
        foreach (var c in module.GetOutputColumns("detection")) dirty.Add("detection:" + c);
        foreach (var c in module.GetOutputColumns("image")) dirty.Add("image:" + c);
    }
    return result;
}

// Combined GS-HOTA over several sequences, as TrackEval's combine_sequences: evaluating the
// concatenated frames with per-sequence identity spaces sums TP/FN/FP per alpha and weights
// AssA/LocA by TP — the same aggregation, without touching the metric.
int EvalCombine(string output, string[] inputs)
{
    // per-sequence metrics are recomputed from the dumped detections, so a metric change re-scores
    // old runs without running the pipeline again
    var dumps = inputs.Select(f => JsonSerializer.Deserialize<EvalDump>(File.ReadAllText(f))!)
        .Select(d => d with
        {
            Frames = d.Gt.Count,
            Full = GsHota.Evaluate(d.Gt, d.Pred),
            NoJersey = GsHota.Evaluate(d.Gt, d.Pred, new GsHotaOptions { UseJerseyNumbers = false }),
            PositionOnly = GsHota.Evaluate(d.Gt, d.Pred,
                new GsHotaOptions { UseJerseyNumbers = false, UseTeams = false, UseRoles = false }),
        })
        .ToList();
    var gtFrames = new List<IReadOnlyList<GsDetection>>();
    var predFrames = new List<IReadOnlyList<GsDetection>>();
    for (var s = 0; s < dumps.Count; s++)
    {
        var offset = (s + 1) * 1_000_000_000L;
        gtFrames.AddRange(dumps[s].Gt.Select(f => (IReadOnlyList<GsDetection>)f.Select(d => d with { TrackId = offset + d.TrackId }).ToList()));
        predFrames.AddRange(dumps[s].Pred.Select(f => (IReadOnlyList<GsDetection>)f.Select(d => d with { TrackId = offset + d.TrackId }).ToList()));
    }
    var full = GsHota.Evaluate(gtFrames, predFrames);
    var noJersey = GsHota.Evaluate(gtFrames, predFrames, new GsHotaOptions { UseJerseyNumbers = false });
    var positionOnly = GsHota.Evaluate(gtFrames, predFrames,
        new GsHotaOptions { UseJerseyNumbers = false, UseTeams = false, UseRoles = false });

    Console.WriteLine($"GS-HOTA combined ({dumps.Count} sequences, {gtFrames.Count} frames)");
    Console.WriteLine($"  full (role+team+jersey): HOTA={full.Hota:P1}  DetA={full.DetA:P1}  AssA={full.AssA:P1}  LocA={full.LocA:P1}");
    Console.WriteLine($"  without jersey numbers:  HOTA={noJersey.Hota:P1}  DetA={noJersey.DetA:P1}  AssA={noJersey.AssA:P1}");
    Console.WriteLine($"  position only:           HOTA={positionOnly.Hota:P1}  DetA={positionOnly.DetA:P1}  AssA={positionOnly.AssA:P1}");
    File.WriteAllText(output, JsonSerializer.Serialize(new { Full = full, NoJersey = noJersey, PositionOnly = positionOnly }));

    // markdown table next to the json: one row per sequence, arithmetic mean, combined
    static string Pct(double v) => (100 * v).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
    static string Row(string name, string frames, HotaResult f, HotaResult n, HotaResult p) =>
        $"| {name} | {frames} | {Pct(f.Hota)} | {Pct(f.DetA)} | {Pct(f.AssA)} | {Pct(f.LocA)} " +
        $"| {Pct(n.Hota)} | {Pct(n.DetA)} | {Pct(n.AssA)} | {Pct(p.Hota)} | {Pct(p.DetA)} | {Pct(p.AssA)} |";
    static HotaResult Mean(IEnumerable<HotaResult> rs) => new()
    {
        Hota = rs.Average(r => r.Hota), DetA = rs.Average(r => r.DetA),
        AssA = rs.Average(r => r.AssA), LocA = rs.Average(r => r.LocA), HotaPerAlpha = [],
    };
    var md = new System.Text.StringBuilder();
    md.AppendLine("| sekwencja | klatki | pełny HOTA | pełny DetA | pełny AssA | pełny LocA | bez nr HOTA | bez nr DetA | bez nr AssA | pozycja HOTA | pozycja DetA | pozycja AssA |");
    md.AppendLine("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
    foreach (var d in dumps)
        md.AppendLine(Row(d.Video, d.Frames.ToString(), d.Full, d.NoJersey, d.PositionOnly));
    md.AppendLine(Row("**średnia sekwencji**", "", Mean(dumps.Select(d => d.Full)),
        Mean(dumps.Select(d => d.NoJersey)), Mean(dumps.Select(d => d.PositionOnly))));
    md.AppendLine(Row("**łącznie (jak TrackEval)**", gtFrames.Count.ToString(), full, noJersey, positionOnly));
    File.WriteAllText(Path.ChangeExtension(output, ".md"), md.ToString());
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

record EvalDump(string Video, int Frames, HotaResult Full, HotaResult NoJersey, HotaResult PositionOnly,
    List<long> ImageIds, List<IReadOnlyList<GsDetection>> Gt, List<IReadOnlyList<GsDetection>> Pred);

/// <summary>Reports on the calling thread, unlike <see cref="Progress{T}"/>, which posts to the thread
/// pool and could print lines out of order.</summary>
sealed class InlineProgress(Action<TrackingProgress> report) : IProgress<TrackingProgress>
{
    public void Report(TrackingProgress value) => report(value);
}
