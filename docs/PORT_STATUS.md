# Port status

How each part of the Python baseline maps to the .NET code.

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
| MMOCR (jersey numbers) | `MmocrOnnx` + `DbPostProcessor` — identical logic; weights: PP-OCRv4 (DB det + CTC rec) instead of DBNet-icdar15+SAR, because mmcv/mmdeploy requires py≤3.11/torch≤2.1 (export: `tools/export/export_ppocr.sh`; SAR route: `tools/export/export_mmocr.py`) | ✅ (substitute weights) |
| NBJW calibration (keypoints → homography) | `NbjwCalibKeypoints` (ONNX) + `HeatmapDecoder` + `KeypointCompleter` + `FrameByFrameCalib` (RANSAC) + `NbjwCalib` (`bbox_pitch`) | ✅ |
| Camera parameters (`get_cam_params` + `heuristic_voting`) | `FrameByFrameCalib.GetCamParams`/`HeuristicVoting` (multi-plane `calibrateCamera`, pan/tilt/roll) → `parameters` column | ✅ |
| `MajorityVoteTracklet` (role + number per tracklet) | `PitchWise.Tracking.Wrappers.MajorityVoteTracklet` | ✅ |
| `TrackletTeamSideLabeling` | pure-logic port | ✅ |
| Visualization (ellipses, minimap, pitch lines) | `GameStateVisualizer` + `GameStateVideoRenderer` (mp4) + original `Radar.png` | ✅ |
| SoccerNet-GSR dataset (`Labels-GameState.json`) | `PitchWise.GameState.Dataset.SoccerNetGameState` | ✅ |
| Hydra (`soccernet.yaml`) | `configs/soccernet.yaml` + `PipelineConfig` (YamlDotNet) | ✅ |
| GS-HOTA (evaluation) | `PitchWise.Tracking.Eval.Hota` (HOTA core from TrackEval) + `PitchWise.GameState.Evaluation.GsHota` (Gaussian kernel σ=2.5 + role/team/number gating) | ✅ |

The neural models run through **Microsoft.ML.OnnxRuntime**; see [MODELS.md](MODELS.md).

