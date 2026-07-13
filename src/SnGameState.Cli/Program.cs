using Microsoft.Extensions.Logging;
using SnGameState.Calibration;
using SnGameState.Dataset;
using SnGameState.Detection;
using SnGameState.Jersey;
using SnGameState.Reid;
using SnGameState.Team;
using SnGameState.Track;
using SnGameState.Visualization;
using TrackLab.Datastruct;
using TrackLab.Engine;
using TrackLab.Pipeline;
using TrackLab.Wrappers;

using var loggerFactory = LoggerFactory.Create(b => b.AddSimpleConsole());
var log = loggerFactory.CreateLogger("sn-gamestate");

return args switch
{
    ["render", var dataset, var split, var video, var output] => Render(dataset, split, video, output),
    ["track", var video, var output, .. var rest] => Track(video, output,
        rest.Length > 0 ? rest[0] : "pretrained_models",
        rest.Length > 1 ? int.Parse(rest[1]) : 0,
        rest.Length > 2 ? double.Parse(rest[2]) : 0),
    _ => Usage(),
};

int Usage()
{
    Console.WriteLine("""
        usage:
          render <dataset_path> <split> <video_name> <out.mp4>   render dataset ground truth
          track <video.mp4> <out.mp4> [models_dir] [max_frames] [start_seconds]  run the pipeline on any video
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

int Track(string videoPath, string output, string modelsDir, int maxFrames, double startSeconds = 0)
{
    string Model(string name) => Path.Combine(modelsDir, name);
    bool Has(params string[] names) => names.All(n => File.Exists(Model(n)));

    if (!Has("yolo11m.onnx"))
    {
        Console.WriteLine($"missing {Model("yolo11m.onnx")} — export it with tools/export/export_yolo.py");
        return 1;
    }

    var framesDir = Path.Combine(
        Path.GetDirectoryName(Path.GetFullPath(output))!,
        $"frames-{Path.GetFileNameWithoutExtension(videoPath)}");
    log.LogInformation("Extracting frames of {Video} to {Dir}", videoPath, framesDir);
    var set = ExternalVideo.Load(videoPath, framesDir, maxFrames: maxFrames, startSeconds: startSeconds);
    var videoId = set.VideoMetadatas.Rows.First().Get<long>("id");
    log.LogInformation("{N} frames", set.ImageMetadatas.Count);

    var hasReid = Has("prtreid-soccernet-baseline.onnx");
    var hasCalib = Has("nbjw_kp.onnx", "nbjw_lines.onnx");
    var hasOcr = Has("mmocr_det.onnx", "mmocr_rec.onnx", "mmocr_charset.txt");
    var imWidth = (int)set.VideoMetadatas.Rows.First().Get<long>("im_width");
    var imHeight = (int)set.VideoMetadatas.Rows.First().Get<long>("im_height");

    var modules = new List<Module> { new YoloOnnxDetector(Model("yolo11m.onnx"), minConfidence: 0.4f) };
    if (hasReid)
        modules.Add(new PrtReidOnnx(Model("prtreid-soccernet-baseline.onnx")));
    modules.Add(new StrongSortModule());
    if (hasCalib)
    {
        modules.Add(new NbjwCalibKeypoints(Model("nbjw_kp.onnx"), Model("nbjw_lines.onnx")));
        modules.Add(new NbjwCalib(imWidth, imHeight));
    }
    if (hasOcr)
        modules.Add(new MmocrOnnx(Model("mmocr_det.onnx"), Model("mmocr_rec.onnx"), Model("mmocr_charset.txt")));

    var voteAttributes = new List<string>();
    if (hasOcr) voteAttributes.Add("jersey_number");
    if (hasReid) voteAttributes.Add("role");
    if (voteAttributes.Count > 0)
        modules.Add(new MajorityVoteTracklet(voteAttributes.ToArray()));
    if (hasReid)
    {
        modules.Add(new TrackletTeamClustering());
        if (hasCalib)
            modules.Add(new TrackletTeamSideLabeling());
    }

    var pipeline = new Pipeline(modules, log);
    log.LogInformation("reid={Reid} calibration={Calib} ocr={Ocr}", hasReid, hasCalib, hasOcr);

    var state = new TrackerState(set);
    var engine = new OfflineTrackingEngine(pipeline, state, log);
    engine.TrackDataset();

    log.LogInformation("Rendering to {Output}", output);
    using var renderer = new GameStateVideoRenderer();
    renderer.RenderVideo(state, videoId, output);
    log.LogInformation("Done");
    return 0;
}
