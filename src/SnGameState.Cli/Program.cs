using Microsoft.Extensions.Logging;
using SnGameState.Dataset;
using SnGameState.Jersey;
using SnGameState.Team;
using SnGameState.Visualization;
using TrackLab.Datastruct;
using TrackLab.Pipeline;

using var loggerFactory = LoggerFactory.Create(b => b.AddSimpleConsole());
var log = loggerFactory.CreateLogger("sn-gamestate");

// Odpowiednik `tracklab -cn soccernet`. Na razie dwie komendy:
//   render <dataset_path> <split> <video_name> <out.mp4>  — render ground truth na mp4 (smoke test całości)
//   pipeline — placeholder pełnego pipeline'u (wymaga modeli ONNX, patrz tools/export)
if (args.Length >= 1 && args[0] == "render")
{
    if (args.Length != 5)
    {
        Console.WriteLine("usage: render <dataset_path> <split> <video_name> <out.mp4>");
        return 1;
    }
    var (datasetPath, split, videoName, output) = (args[1], args[2], args[3], args[4]);

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

var pipeline = new Pipeline(new Module[]
{
    // new YoloOnnxDetector("pretrained_models/yolo11m.onnx"),
    // new PrtReidOnnx("pretrained_models/prtreid-soccernet-baseline.onnx"),
    // new StrongSortModule(),
    // new NbjwCalibKeypoints("pretrained_models/nbjw_kp.onnx", "pretrained_models/nbjw_lines.onnx"),
    // new NbjwCalib(1920, 1080),
    new TrackletTeamClustering(),
    new TrackletTeamSideLabeling(),
    new VotingTrackletJerseyNumber(),
}, log);

log.LogInformation("dotnet-sn-gamestate — port SoccerNet Game State Reconstruction");
log.LogInformation("Pipeline: {Pipeline}", pipeline.ToString());
log.LogInformation("Modele ONNX wymagane do pełnego przebiegu — patrz tools/export/*.py");
return 0;
