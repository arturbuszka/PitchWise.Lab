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
| PRTReid (reid + rola + drużyna) | ONNX (eksport z prtreid) | 🔜 |
| MMOCR (numery koszulek) | ONNX (detektor tekstu + rozpoznawanie) | 🔜 |
| Kalibracja (TVCalib / NBJW / PnLCalib) | ONNX (keypointy linii) + homografia (OpenCvSharp) | 🔜 |
| `TrackletTeamSideLabeling` | port czystej logiki | 🔜 |
| Wizualizacja (elipsy, minimapa) | OpenCvSharp | 🔜 |
| Hydra (`soccernet.yaml`) | YamlDotNet + kompozycja configów | 🔜 |
| GS-HOTA (ewaluacja) | port z TrackEval | 🔜 |

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
```
