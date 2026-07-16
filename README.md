# PitchWise — Vision & Game State

The vision and game-state engine of **PitchWise**, a football analytics platform. From a raw
broadcast video (1080p, main camera) the pipeline reconstructs the *game state*: the position of
every person in pitch coordinates, plus their role (player / goalkeeper / referee), team
(left / right) and jersey number. The tracking quality metric is GS-HOTA.

This repository is the vision half of PitchWise. A separate platform (streaming, highlight cutting,
etc.) is built alongside it and consumes the world-state this engine produces.

It is a .NET port of the SoccerNet
[Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction) baseline
([sn-gamestate](https://github.com/SoccerNet/sn-gamestate) +
[TrackLab](https://github.com/TrackingLaboratory/tracklab)).

Licensed under **GPL-3.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Architecture

| Original (Python) | .NET | Status |
|---|---|---|
| `tracklab.pipeline.Module` / `Pipeline` | `PitchWise.Tracking.Pipeline.Module` / `Pipeline` (I/O column validation) | ✅ |
| `ImageLevel/DetectionLevel/VideoLevelModule` | equivalent abstract classes | ✅ |
| `pandas.DataFrame` (detections / metadata) | `PitchWise.Tracking.Datastruct.DataFrame`/`Series` (dynamic columns, merge by index) | ✅ |
| `TrackerState`, `TrackingDataset` | `TrackerState`, `TrackingSet` (JSON persistence) | ✅ (simplified) |
| `engine.offline.OfflineTrackingEngine` | `PitchWise.Tracking.Engine.OfflineTrackingEngine` | ✅ |
| `utils.coordinates`, `attribute_voting` | `PitchWise.Tracking.Utils.Coordinates`, `AttributeVoting` | ✅ |
| YOLOv11 (ultralytics) | `YoloOnnxDetector` — ONNX Runtime + letterbox + NMS | ✅ (requires model export) |
| `TrackletTeamClustering` (sklearn KMeans) | `TrackletTeamClustering` + own KMeans (k-means++) | ✅ |
| `VotingTrackletJerseyNumber` | `VotingTrackletJerseyNumber` | ✅ |
| StrongSORT (tracking, bpbreid_strong_sort) | separate MIT package [`StrongSort.Net`](https://github.com/arturbuszka/StrongSort.NET) (zero deps) + adapter `PitchWise.GameState.Track.StrongSortModule` | ✅ |
| PRTReid (reid + role + team) | `PrtReidOnnx` — ONNX (export: `tools/export/export_prtreid.py`) | ✅ |
| MMOCR (jersey numbers) | `MmocrOnnx` + `DbPostProcessor` — identical logic; weights: PP-OCRv4 (DB det + CTC rec) instead of DBNet-icdar15+SAR, because mmcv/mmdeploy requires py≤3.11/torch≤2.1 (SAR export: `tools/export/export_mmocr.py`) | ✅ (substitute weights) |
| NBJW calibration (keypoints → homography) | `NbjwCalibKeypoints` (ONNX) + `HeatmapDecoder` + `KeypointCompleter` + `FrameByFrameCalib` (RANSAC) + `NbjwCalib` (`bbox_pitch`) | ✅ |
| Camera parameters (`get_cam_params` + `heuristic_voting`) | `FrameByFrameCalib.GetCamParams`/`HeuristicVoting` (multi-plane `calibrateCamera`, pan/tilt/roll) → `parameters` column | ✅ |
| `MajorityVoteTracklet` (role + number per tracklet) | `PitchWise.Tracking.Wrappers.MajorityVoteTracklet` | ✅ |
| `TrackletTeamSideLabeling` | pure-logic port | ✅ |
| Visualization (ellipses, minimap, pitch lines) | `GameStateVisualizer` + `GameStateVideoRenderer` (mp4) + original `Radar.png` | ✅ |
| SoccerNet-GSR dataset (`Labels-GameState.json`) | `PitchWise.GameState.Dataset.SoccerNetGameState` | ✅ |
| Hydra (`soccernet.yaml`) | `configs/soccernet.yaml` + `PipelineConfig` (YamlDotNet) | ✅ |
| GS-HOTA (evaluation) | `PitchWise.Tracking.Eval.Hota` (HOTA core from TrackEval) + `PitchWise.GameState.Evaluation.GsHota` (Gaussian kernel σ=2.5 + role/team/number gating) | ✅ |

The neural models (YOLO, PRTReid, MMOCR, calibration) run through **Microsoft.ML.OnnxRuntime** — the
PyTorch weights must be exported to ONNX once (scripts under `tools/export`).

## Structure

- `src/PitchWise.Tracking` — tracking-framework core (port of `tracklab`)
- `src/PitchWise.GameState` — football modules (port of `sn_gamestate`)
- `src/PitchWise.Engine` — dependency-free football rule/world-state engine
- `src/PitchWise.Cli` — entry point (equivalent of `tracklab -cn soccernet`)

StrongSORT tracking is provided by the separate MIT-licensed
[StrongSort.Net](https://github.com/arturbuszka/StrongSort.NET) package, referenced via NuGet.

## Build

```
dotnet build PitchWise.slnx
dotnet run --project src/PitchWise.Cli
# render ground truth to mp4 (ellipses + minimap):
dotnet run --project src/PitchWise.Cli -- render <SoccerNetGS_path> valid SNGS-021 out.mp4
# full pipeline on any video (requires ONNX models in pretrained_models/):
dotnet run --project src/PitchWise.Cli -- track match.mp4 out.mp4 [models_dir] [max_frames] [start_seconds]
# pipeline + GS-HOTA metric vs dataset ground truth:
dotnet run --project src/PitchWise.Cli -- eval <SoccerNetGS_path> valid SNGS-021 [models_dir] [max_frames]
```

Module parameters are configured in `configs/soccernet.yaml` (the Hydra equivalent; use the `-c other.yaml`
flag to switch). Modules with `enabled: auto` turn on when their ONNX models are present in `models_dir`.

`track` assembles the pipeline from the modules whose models are available: `yolo11m.onnx` (required) →
`prtreid-soccernet-baseline.onnx` (teams / roles) → StrongSORT → `nbjw_kp/lines.onnx`
(homography + minimap) → `mmocr_det/rec.onnx` (numbers). Export: `tools/export/*.py`.
