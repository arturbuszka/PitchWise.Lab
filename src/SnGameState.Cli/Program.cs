using Microsoft.Extensions.Logging;
using SnGameState.Jersey;
using SnGameState.Team;
using TrackLab.Pipeline;

using var loggerFactory = LoggerFactory.Create(b => b.AddSimpleConsole());
var log = loggerFactory.CreateLogger("sn-gamestate");

// Odpowiednik `tracklab -cn soccernet` — na razie składamy pipeline w kodzie;
// docelowo konfiguracja YAML (odpowiednik Hydry) wybierze moduły per etap.
var pipeline = new Pipeline(new Module[]
{
    // new YoloOnnxDetector("pretrained_models/yolo11m.onnx"),  // wymaga wyeksportowanego modelu ONNX
    new TrackletTeamClustering(),
    new VotingTrackletJerseyNumber(),
}, log);

log.LogInformation("dotnet-sn-gamestate — port SoccerNet Game State Reconstruction");
log.LogInformation("Pipeline: {Pipeline}", pipeline.ToString());
