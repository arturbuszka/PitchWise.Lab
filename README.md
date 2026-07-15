# dotnet-sn-gamestate

Port 1:1 [SoccerNet Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction)
([sn-gamestate](https://github.com/SoccerNet/sn-gamestate) + [TrackLab](https://github.com/TrackingLaboratory/tracklab)) do .NET.

Z surowego wideo transmisji (1080p, główna kamera) pipeline odtwarza „stan gry”:
pozycje wszystkich osób we współrzędnych boiska + rola (player/goalkeeper/referee) +
drużyna (left/right) + numer na koszulce. Metryka: GS-HOTA.

## Architektura

| Python | .NET | Status |
|---|---|---|
| `tracklab.pipeline.Module` / `Pipeline` | `TrackLab.Pipeline.Module` / `Pipeline` (walidacja kolumn we/wy) | ✅ |
| `ImageLevel/DetectionLevel/VideoLevelModule` | analogiczne klasy abstrakcyjne | ✅ |
| `pandas.DataFrame` (detekcje/metadane) | `TrackLab.Datastruct.DataFrame`/`Series` (dynamiczne kolumny, merge po indeksie) | ✅ |
| `TrackerState`, `TrackingDataset` | `TrackerState`, `TrackingSet` (persystencja JSON) | ✅ (uproszczone) |
| `engine.offline.OfflineTrackingEngine` | `TrackLab.Engine.OfflineTrackingEngine` | ✅ |
| `utils.coordinates`, `attribute_voting` | `TrackLab.Utils.Coordinates`, `AttributeVoting` | ✅ |
| YOLOv11 (ultralytics) | `YoloOnnxDetector` — ONNX Runtime + letterbox + NMS | ✅ (wymaga eksportu modelu) |
| `TrackletTeamClustering` (sklearn KMeans) | `TrackletTeamClustering` + własny KMeans (k-means++) | ✅ |
| `VotingTrackletJerseyNumber` | `VotingTrackletJerseyNumber` | ✅ |
| StrongSORT (tracking, bpbreid_strong_sort) | osobna biblioteka **`src/StrongSort`** (pakiet `StrongSort.Net`, zero zależności) + adapter `SnGameState.Track.StrongSortModule` | ✅ |
| PRTReid (reid + rola + drużyna) | `PrtReidOnnx` — ONNX (eksport: `tools/export/export_prtreid.py`) | ✅ |
| MMOCR (numery koszulek) | `MmocrOnnx` + `DbPostProcessor` — logika 1:1; wagi: PP-OCRv4 (det DB + rec CTC) zamiast DBNet-icdar15+SAR, bo mmcv/mmdeploy wymaga py≤3.11/torch≤2.1 (eksport SAR: `tools/export/export_mmocr.py`) | ✅ (wagi zastępcze) |
| Kalibracja NBJW (keypointy → homografia) | `NbjwCalibKeypoints` (ONNX) + `HeatmapDecoder` + `KeypointCompleter` + `FrameByFrameCalib` (RANSAC) + `NbjwCalib` (`bbox_pitch`) | ✅ |
| Parametry kamery (`get_cam_params` + `heuristic_voting`) | `FrameByFrameCalib.GetCamParams`/`HeuristicVoting` (multi-plane `calibrateCamera`, pan/tilt/roll) → kolumna `parameters` | ✅ |
| `MajorityVoteTracklet` (rola+numer per tracklet) | `TrackLab.Wrappers.MajorityVoteTracklet` | ✅ |
| `TrackletTeamSideLabeling` | port czystej logiki | ✅ |
| Wizualizacja (elipsy, minimapa, linie boiska) | `GameStateVisualizer` + `GameStateVideoRenderer` (mp4) + oryginalny `Radar.png` | ✅ |
| Dataset SoccerNet-GSR (`Labels-GameState.json`) | `SnGameState.Dataset.SoccerNetGameState` | ✅ |
| Hydra (`soccernet.yaml`) | `configs/soccernet.yaml` + `PipelineConfig` (YamlDotNet) | ✅ |
| GS-HOTA (ewaluacja) | `TrackLab.Eval.Hota` (rdzeń HOTA z TrackEval) + `SnGameState.Evaluation.GsHota` (jądro gaussowskie σ=2.5 + bramkowanie rola/drużyna/numer) | ✅ |

Modele neuronowe (YOLO, PRTReid, MMOCR, kalibracja) biegną przez **Microsoft.ML.OnnxRuntime** —
wagi PyTorch trzeba jednorazowo wyeksportować do ONNX (skrypty będą w `tools/export`).

## Struktura

- `src/StrongSort` — samodzielna implementacja StrongSORT (NSA Kalman, EMA appearance, Hungarian) — publikowalna jako NuGet `StrongSort.Net` (`dotnet pack src/StrongSort`)
- `src/TrackLab` — rdzeń frameworku trackingu (port `tracklab`)
- `src/SnGameState` — moduły piłkarskie (port `sn_gamestate`)
- `src/SnGameState.Cli` — punkt wejścia (odpowiednik `tracklab -cn soccernet`)
- `reference/` — sklonowane źródła Pythona (tylko do wglądu, poza kompilacją)

## Build

```
dotnet build SnGameState.slnx
dotnet run --project src/SnGameState.Cli
# render ground truth na mp4 (elipsy + minimapa):
dotnet run --project src/SnGameState.Cli -- render <ścieżka_SoccerNetGS> valid SNGS-021 out.mp4
# pełny pipeline na dowolnym wideo (wymaga modeli ONNX w pretrained_models/):
dotnet run --project src/SnGameState.Cli -- track mecz.mp4 out.mp4 [models_dir] [max_frames] [start_seconds]
# pipeline + metryka GS-HOTA vs ground truth datasetu:
dotnet run --project src/SnGameState.Cli -- eval <ścieżka_SoccerNetGS> valid SNGS-021 [models_dir] [max_frames]
```

Parametry modułów konfiguruje `configs/soccernet.yaml` (odpowiednik Hydry; flaga `-c inny.yaml`).
Moduły z `enabled: auto` włączają się, gdy ich modele ONNX są w `models_dir`.

`track` składa pipeline z modułów, dla których są modele: `yolo11m.onnx` (wymagany) →
`prtreid-soccernet-baseline.onnx` (drużyny/role) → StrongSORT → `nbjw_kp/lines.onnx`
(homografia + minimapa) → `mmocr_det/rec.onnx` (numery). Eksport: `tools/export/*.py`.
