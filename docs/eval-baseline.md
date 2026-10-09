# Baseline warstwy wizji — SoccerNet-GSR

Pomiar czysto diagnostyczny: nic w wizji nie było poprawiane. Tabelę poniżej (blok między markerami `eval:begin` i `eval:end`)
nadpisuje skrypt `tools/eval/run_eval.sh` (Linux) albo `tools/eval/run_eval.ps1` (Windows/pwsh). Puszcza się go po każdej zmianie w wizji.
Reszta pliku jest pisana ręcznie i skrypt jej nie rusza.

## Co jest mierzone

- **Zbiór:** SoccerNet-GSR (`gamestate-2024`), split `valid`, v1.3 (`valid.zip`, sha256 `d60e517e…`), rozpakowany do `data/SoccerNetGS/valid/`.
- **Sekwencje:** 5 pełnych klipów po 750 klatek, wybrane raz ze stałym seedem. Lista i sposób wyboru są w [`eval-sequences.txt`](eval-sequences.txt).
- **Komenda na sekwencję:** `dotnet run --project src/PitchWise.Cli -c Release -- eval data/SoccerNetGS valid <SNGS-xxx> --dump <plik.json>`.
  Opcja `--dump` zapisuje metryki i oceniane detekcje. `eval-combine` liczy z nich wiersz średniej i wynik łączny.
- **Trzy warianty GS-HOTA** (ta sama predykcja, coraz łagodniej oceniana):
  - *pełny*: musi się zgadzać pozycja, rola, drużyna i numer;
  - *bez numerów*: bez numeru;
  - *sama pozycja*: tylko pozycja i ciągłość ID.
- **Średnia:** dwa wiersze.
  - *Średnia sekwencji* to zwykła średnia arytmetyczna z pięciu wierszy.
  - *Łącznie* to agregacja jak w TrackEval: TP, FN i FP sumowane dla każdego α, a AssA i LocA ważone liczbą TP.
    Oficjalne liczby SoccerNet liczy się tym drugim sposobem.

### Moduły pipeline'u, które były włączone

Wszystkie modele z `configs/soccernet.yaml` są w `pretrained_models/`, więc wszystkie moduły `enabled: auto` się włączyły:

`YoloOnnxDetector → PrtReidOnnx → StrongSortModule → NbjwCalibKeypoints → NbjwCalib → MmocrOnnx → MajorityVoteTracklet → TrackletTeamClustering → TrackletTeamSideLabeling`

| moduł | plik | uwagi |
|---|---|---|
| detekcja | `yolo11m.onnx` | ogólny model COCO (klasa `person`); oryginał używa tego samego |
| ReID, rola, drużyna | `prtreid-soccernet-baseline.onnx` | trenowany na SoccerNet |
| kalibracja | `nbjw_kp.onnx`, `nbjw_lines.onnx` | NBJW, trenowany na SoccerNet |
| numery (OCR) | `mmocr_det.onnx`, `mmocr_rec.onnx`, `mmocr_charset.txt` | **zamiennik:** PP-OCRv4 z PaddleOCR (`ch_PP-OCRv4_det_infer` i `en_PP-OCRv4_rec_infer`, przekonwertowane `paddle2onnx 1.3.1`, opset 14; charset to `en_dict.txt` + spacja, 96 znaków). Oryginał używa MMOCR DBNet + SAR. Tych plików wcześniej nie było w repo, dodane 2026-10-08 na potrzeby tego pomiaru |

**Uwaga o OCR.** Sprawdziłem modele OCR niezależnie od C#: w Pythonie, ten sam ONNX, na wycinkach zawodników z GT.
Na całym SNGS-022 poprawny numer jest w odczytach 35 z 200 wycinków; często są to dwucyfrowe numery, np. „14”. W szerokim planie
(pierwsze 30 klatek SNGS-022) trafienia spadają do 14 z 289, a odczyty to głównie pojedyncze cyfry. Tak samo zachowuje się wynik C#.
Słabe numery to więc cecha tego modelu na tym materiale, a nie błąd integracji.

### Środowisko i dwie zmiany potrzebne, żeby pomiar w ogóle przeszedł

- **CUDA:** ONNX Runtime (`Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.24.4) potrzebuje CUDA 12 i cuDNN 9 z `~/cuda12-libs`
  (wheele NVIDIA, cuDNN 9.10.2). Skrypty same dodają ten katalog do `LD_LIBRARY_PATH`; inny katalog można podać zmienną `CUDA_LIBS`.
  Bez tych bibliotek `OnnxSessions` po cichu przechodzi na CPU. Skrypt to wykrywa, wypisuje ostrzeżenie i zapisuje
  provider każdego modelu w bloku metadanych.
- **`onnx.low_vram: true`:** na GTX 1660 Ti (6 GB) pełny pipeline kończył się błędem OOM, powtarzalnie na SNGS-028.
  Każda sesja ONNX trzyma własną arenę pamięci do końca procesu. Dodatkowo cuDNN rezerwuje maksymalny workspace konwolucji,
  przez co NBJW zajmował ok. 4.9 GB.
  Tryb `low_vram` w `OnnxSessions` robi trzy rzeczy: ustawia `arena_extend_strategy=kSameAsRequested` i `cudnn_conv_use_max_workspace=0`
  oraz zwalnia arenę po każdym `Run` (`memory.enable_memory_arena_shrinkage`).
  Szczyt VRAM spadł z 5.75 GB do ok. 2.1 GB. Wyniki są **bit w bit identyczne**: zrzut SNGS-022 jest taki sam jak z biegu bez tego trybu.
  Na większej karcie wystarczy ustawić `low_vram: false`.
- **Czas:** ok. 45 min na sekwencję, z czego ok. 2.7 s na klatkę to OCR. Detektor OCR dostaje wycinki o różnych rozmiarach,
  a cuDNN dostraja się do każdego kształtu osobno.

## Jak mierzyć

| cel | komenda (bash; w PowerShell analogiczne parametry `-Resume`, `-Rescore`, `-FromCache`, `-Rerun`) | czas |
|---|---|---|
| pełny pomiar (np. po zmianie modelu) | `tools/eval/run_eval.sh` | ok. 4 h |
| dokończenie przerwanego biegu | `RESUME=data/eval-runs/<run> tools/eval/run_eval.sh` | to, czego brakuje |
| eksperyment na kopii configu, bez ruszania tej tabeli | dodaj `CONFIG=<plik.yaml> DOC=<plik.md>` | jw. |
| tylko przeliczenie metryki | `RESCORE=data/eval-runs/<run> tools/eval/run_eval.sh` | sekundy |
| zmiana w trackerze / głosowaniu / drużynach | `FROM_CACHE=data/eval-runs/<run> RERUN=StrongSortModule tools/eval/run_eval.sh` | ok. 2 min |

Każdy pełny bieg zapisuje stan pipeline'u (wszystkie kolumny) w `<run>/cache/`. `FROM_CACHE` wczytuje go i uruchamia
tylko moduły z `RERUN` oraz te, które czytają ich kolumny wyjściowe (wyznaczane z `InputColumns` i `OutputColumns`).
Dla `StrongSortModule` są to: `StrongSortModule → MajorityVoteTracklet → TrackletTeamClustering → TrackletTeamSideLabeling`.
Wynik jest identyczny z pełnym biegiem: sprawdzone na SNGS-028 (60 klatek), zrzut bit w bit ten sam.
Jeśli od zapisania cache zmieniła się sekcja configu albo plik modelu dowolnego pomijanego modułu, bieg kończy się błędem
z prośbą o dopisanie tego modułu do `RERUN`. Dzięki temu nie da się po cichu użyć nieaktualnych wyników.

<!-- eval:begin (generated by tools/eval/run_eval.sh|.ps1 — do not edit by hand) -->
Pomiar: 2026-10-09 01:53, split `valid`, 5 sekwencji (pełne, bez max_frames), wyniki w %.

| sekwencja | klatki | pełny HOTA | pełny DetA | pełny AssA | pełny LocA | bez nr HOTA | bez nr DetA | bez nr AssA | pozycja HOTA | pozycja DetA | pozycja AssA |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SNGS-022 | 722 | 16.0 | 4.7 | 54.8 | 95.5 | 26.4 | 21.7 | 33.5 | 57.9 | 82.2 | 40.9 |
| SNGS-028 | 676 | 29.7 | 20.7 | 42.9 | 91.9 | 44.1 | 55.6 | 35.1 | 50.5 | 69.1 | 37.0 |
| SNGS-038 | 749 | 53.5 | 44.6 | 64.4 | 90.0 | 57.8 | 57.9 | 57.8 | 60.3 | 65.4 | 55.7 |
| SNGS-079 | 740 | 16.8 | 11.2 | 25.4 | 93.5 | 28.1 | 35.8 | 22.2 | 38.1 | 59.3 | 24.6 |
| SNGS-086 | 750 | 35.2 | 30.0 | 41.3 | 92.5 | 43.4 | 57.8 | 32.6 | 45.4 | 66.2 | 31.2 |
| **średnia sekwencji** |  | 30.3 | 22.2 | 45.8 | 92.7 | 40.0 | 45.8 | 36.2 | 50.5 | 68.5 | 37.9 |
| **łącznie (jak TrackEval)** | 3637 | 31.9 | 21.1 | 48.1 | 91.8 | 40.5 | 44.5 | 37.0 | 50.3 | 67.6 | 37.4 |

- **Commit:** `ccc1d92` + niezacommitowane zmiany:
  - `AM docs/eval-baseline.md`
  - `A  docs/eval-sequences.txt`
  - ` M src/PitchWise.Cli/Program.cs`
  - ` M src/PitchWise.GameState/Configuration/PipelineConfig.cs`
  - ` M src/PitchWise.GameState/Dataset/SoccerNetGameState.cs`
  - ` M src/PitchWise.GameState/Evaluation/GsHota.cs`
  - ` M src/PitchWise.Tracking/Engine/OfflineTrackingEngine.cs`
  - ` M tests/PitchWise.GameState.Tests/GsHotaTests.cs`
  - `AM tools/eval/run_eval.ps1`
  - `AM tools/eval/run_eval.sh`
- **Provider ONNX (z logów):**
  - ONNX mmocr_det.onnx: running on CUDA (device 0)
  - ONNX mmocr_rec.onnx: running on CUDA (device 0)
  - ONNX nbjw_kp.onnx: running on CUDA (device 0)
  - ONNX nbjw_lines.onnx: running on CUDA (device 0)
  - ONNX prtreid-soccernet-baseline.onnx: running on CUDA (device 0)
  - ONNX yolo11m.onnx: running on CUDA (device 0)
- **GPU:** NVIDIA GeForce GTX 1660 Ti with Max-Q Design, 615.71.09
- **Surowe wyniki:** `data/eval-runs/20261008-215833/`
- **Modele (sha256, pierwsze 12 znaków):**
  - `mmocr_det.onnx` 69ce850fec74
  - `mmocr_rec.onnx` 02672d591fee
  - `nbjw_kp.onnx` e6ee8ae996f1
  - `nbjw_lines.onnx` 7d8d057ca64a
  - `prtreid-soccernet-baseline.onnx` 888b92b64c5f
  - `yolo11m.onnx` 72e21f2178d3
  - `nbjw_kp.onnx.data` fafb55681468
  - `nbjw_lines.onnx.data` 1acfa38e5583
  - `mmocr_charset.txt` aab362a7748f
- **Config (`configs/soccernet.yaml`):**

```yaml
# Default pipeline configuration (the Hydra equivalent) for the .NET port.
# Modules with "enabled: auto" turn on when their ONNX models exist in models_dir.
models_dir: pretrained_models

onnx:
  provider: cuda   # cpu | dml (DirectML — any DX12 GPU on Windows, no install needed)
  device_id: 0    # CUDA: index among NVIDIA GPUs only (0 = first). DirectML on hybrid laptops: 0 is often the integrated GPU, 1 the discrete one
  low_vram: true  # CUDA on <= 6 GB GPUs: release arena memory between runs (avoids OOM); set false on bigger GPUs

bbox_detector:
  model: yolo11m.onnx
  min_confidence: 0.4
  input_size: 640

reid:
  enabled: auto
  model: prtreid-soccernet-baseline.onnx
  batch_size: 8

track:  # StrongSORT hyperparameters
  ema_alpha: 0.9
  mc_lambda: 0.995
  max_appearance_distance: 0.5
  max_iou_distance: 0.8
  max_age: 300
  n_init: 0
  min_confidence: 0.0

pitch:
  enabled: auto
  kp_model: nbjw_kp.onnx
  lines_model: nbjw_lines.onnx

calibration:
  enabled: auto
  use_prev_homography: true

jersey:
  enabled: auto
  det_model: mmocr_det.onnx
  rec_model: mmocr_rec.onnx
  charset: mmocr_charset.txt
```
<!-- eval:end -->

## Zgodność metryki z oryginałem

Porównanie [`GsHota.cs`](../src/PitchWise.GameState/Evaluation/GsHota.cs), [`Hota.cs`](../src/PitchWise.Tracking/Eval/Hota.cs)
i przygotowania danych w `Eval` ([`Program.cs`](../src/PitchWise.Cli/Program.cs)) z wersjami, które przypina `reference/sn-gamestate/uv.lock`:
`sn-trackeval==0.4.0` (`trackeval/metrics/hota.py`, `trackeval/datasets/soccernet_gs.py`) i
`tracklab==1.3.24` (`SoccerNetGameState.save_for_eval`, `configs/dataset/soccernet_gs.yaml`).

**Rdzeń HOTA jest zgodny.** Zrzuty z `eval --dump` przeliczone oryginalną klasą `trackeval.metrics.HOTA`
dają wyniki C# z dokładnością do 3·10⁻¹⁵ na każdej sekwencji i w każdym wariancie, także w wyniku łącznym.

### Różnice znalezione i wyrównane 2026-10-08

Wszystkie cztery różnice były w tym, **co** trafia do metryki. Po zatwierdzeniu zostały poprawione, więc obecna tabela to już oficjalna GS-HOTA:

| # | Co | Było w PitchWise | Oryginał (teraz także PitchWise) |
|---|---|---|---|
| 1 | Sigma jądra gaussowskiego | 2.5 m | tolerancja `EVAL_DIST_TOL = 5` m → `calculate_sigma(5)` ≈ **2.04 m**; komentarz w yaml tracklaba mówi „sigma=2.5”, ale kod liczy 2.04; wzór 8 w artykule AuxFlow (CVIU 2026) to potwierdza |
| 2 | Numer koszulki u bramkarzy | sprawdzany | sprawdzany tylko dla `role == "player"`; bramkarz musi się zgadzać tylko rolą i drużyną |
| 3 | Oceniane klatki | wszystkie 750 | tylko klatki z `has_labeled_pitch && has_labeled_camera && has_labeled_person`; w tych 5 sekwencjach odpada 0–74 klatek, wszystkie z `has_labeled_camera = false` |
| 4 | Predykcje z `role == "ball"` | liczone jako FP | odrzucane (`IGNORE_BALL` działa po obu stronach) |

Wpływ na baseline (wynik łączny, liczony oryginalnym `trackeval.HOTA` na tych samych zrzutach):

| | pełny HOTA | bez nr HOTA | pozycja HOTA | pozycja DetA |
|---|--:|--:|--:|--:|
| stara metryka PitchWise | 32.0 | 41.6 | 51.3 | 69.4 |
| tylko #1 (sigma) | 30.8 | 39.8 | 49.4 | 66.4 |
| tylko #2 (bramkarz) | 32.7 | 41.6 | 51.3 | 69.4 |
| tylko #3 (klatki) | 32.5 | 42.3 | 52.2 | 70.6 |
| tylko #4 (ball; 3 takie predykcje na ok. 45 tys.) | 32.0 | 41.6 | 51.3 | 69.4 |
| **wszystkie = obecna metryka** | **31.9** | **40.5** | **50.3** | **67.6** |

`eval-combine` przelicza metryki z zapisanych detekcji, więc zmiana metryki nie wymaga ponownego biegu pipeline'u:
`RESCORE=data/eval-runs/<run> tools/eval/run_eval.sh`. Tak powstała obecna tabela: zrzuty z biegu `20261008-175113`
zostały raz przefiltrowane (#3, #4) do katalogu `20261008-175113-rescored`. Nowe biegi filtrują już same.

### Różnice, które zostały, bez wpływu w praktyce

- **LocA, gdy dla danego α TP = 0:** w oryginale wychodzi 1, w C# 0. Ma znaczenie tylko przy α bliskim 0.95 i braku dopasowań.
- **Pusta sekwencja (0 predykcji):** w oryginale LocA = 1, w C# wszystko 0. HOTA w obu przypadkach to 0.
- **Hungarian:** w C# macierz score jest rzutowana na float32, w oryginale to float64 (`scipy`). Na tych danych nie zmieniło to żadnego dopasowania (zgodność 3·10⁻¹⁵).

## Interpretacja (wynik łączny, 5 sekwencji, oficjalna metryka)

Trzy różnice z zadania, w punktach procentowych:

| różnica | HOTA | DetA | AssA |
|---|--:|--:|--:|
| pełny → bez numerów | +8.6 (31.9 → 40.5) | +23.4 (21.1 → 44.5) | −11.1 (48.1 → 37.0) |
| bez numerów → sama pozycja | +9.8 (40.5 → 50.3) | +23.1 (44.5 → 67.6) | +0.4 (37.0 → 37.4) |
| DetA vs AssA (sama pozycja) | | 67.6 | 37.4 → **30.2 pkt różnicy** |

**Największa jest różnica DetA vs AssA: 30.2 pkt.** Wskazuje ona na **asocjację, czyli tracking (StrongSORT z cechami wyglądu z PRTReid)**.
Detekcja z rzutowaniem na boisko działa przyzwoicie (DetA ok. 68%, LocA ok. 93%), gorzej z utrzymaniem tożsamości.
Na 5 sekwencjach jest 532 przewidziane ID na 109 prawdziwych osób, średnio ok. 5 torów na jednego człowieka.
Najgorzej wypada SNGS-079 (159 ID na 24 osoby, AssA 25%). W wariancie „sama pozycja” AssA ≪ DetA w każdej z 5 sekwencji.

Dwie pozostałe różnice dają w HOTA podobnie (8.6 i 9.8 pkt), ale ich DetA mówi więcej:

- **Numery (OCR + głosowanie na tracklet), 23.4 pkt DetA.** Wymóg zgodnego numeru zamienia ponad połowę trafień w pudła.
  AssA przy tym *rośnie*, bo zostają tylko dopasowania z poprawnym numerem, a te mają dłuższe, spójne tory.
  Rozrzut między sekwencjami jest duży: SNGS-022 traci 10.4 pkt HOTA, SNGS-038 tylko 4.3.
- **Rola i drużyna (PRTReid + klasteryzacja drużyn), 23.1 pkt DetA.** Przykład z danych: w SNGS-028 GT ma 409 detekcji bramkarzy,
  a predykcje tylko 43.

**Uzupełnienie (2026-10-09): strata „rola i drużyna” to w dużej mierze zamiana stron drużyn w całym klipie.**
Zgodność strony (lewa/prawa) predykcji z najbliższym zawodnikiem z GT (do 2 m, ta sama klatka):
SNGS-022 **22%** (strony prawie w całości odwrócone), SNGS-079 67%, SNGS-086 87%, SNGS-038 91%, SNGS-028 93%.
Decyzja `TrackletTeamSideLabeling` o tym, która drużyna jest „lewa”, jest niestabilna. Drobna zmiana w trackerze (`max_age 30`)
naprawia SNGS-022 (95%), a psuje SNGS-028 (46%). Taka zamiana przesuwa wariant „bez numerów” pojedynczej sekwencji o 20–35 pkt.
Dopóki to się nie zmieni, warianty „pełny” i „bez numerów” są zdominowane przez ten efekt. Zmiany w trackingu warto więc oceniać
wariantem „sama pozycja”.

Kolejność dla tej bazy: **tracking/asocjacja > numery ≈ rola i drużyna (głównie strona drużyny) > detekcja i kalibracja.**
Przy czym na DetA w wariancie „sama pozycja” (68%) składają się zarówno pominięte osoby (ok. 44 tys. predykcji na ok. 55 tys. GT),
jak i błędy kalibracji. Ten pomiar ich nie rozdziela.

## Eksperymenty

### StrongSORT, jeden parametr naraz (2026-10-09)

Z cache biegu `20261008-215833` (`FROM_CACHE`, `RERUN=StrongSortModule`). Wynik łączny dla 5 sekwencji; baseline to config z tabeli wyżej.

| parametr (baseline) | wartość | pełny HOTA | bez nr HOTA | pozycja HOTA | pozycja AssA |
|---|--:|--:|--:|--:|--:|
| — | baseline | 31.9 | 40.5 | 50.3 | 37.4 |
| `max_appearance_distance` (0.5) | 0.3 / 0.4 / 0.6 / 0.7 | 31.3 / 34.2 / 30.4 / 29.1 | 40.0 / 45.4 / 39.3 / 37.8 | 50.9 / 50.5 / 50.2 / 50.5 | 38.3 / 37.7 / 37.2 / 37.7 |
| `max_iou_distance` (0.8) | 0.6 / 0.7 / 0.9 | 31.3 / 32.6 / 33.9 | 39.7 / 43.3 / 44.5 | 48.0 / 49.9 / 50.5 | 34.1 / 36.9 / 37.7 |
| `ema_alpha` (0.9) | 0.8 / 0.95 | 33.0 / 33.8 | 44.8 / 45.8 | 49.5 / 51.0 | 36.3 / 38.5 |
| `mc_lambda` (0.995) | 0.98 / 0.9 | 32.6 / 31.1 | 43.1 / 42.1 | 50.3 / 49.5 | 37.4 / 36.3 |
| `max_age` (300) | 10 / 20 / 30 / 50 / 100 / 750 | 31.2 / 31.5 / 33.2 / 32.3 / 34.2 / 31.2 | 41.7 / 43.2 / 43.5 / 42.8 / 45.4 / 41.1 | **53.3 / 53.1 / 53.2** / 52.7 / 51.6 / 50.2 | **41.9 / 41.6 / 41.8** / 41.1 / 39.3 / 37.3 |

- Kolumny „pełny” i „bez nr” skaczą niemonotonicznie o ±3–5 pkt. To głównie zamiana stron drużyn (patrz wyżej), a nie jakość trackingu.
- Jedyny spójny efekt to **`max_age` 10–30**: „sama pozycja” 50.3 → ok. 53.2 (+2.9 HOTA, +4.5 AssA), z poprawą **w każdej z 5 sekwencji**
  (np. przy 30: 57.9→62.8, 50.5→54.6, 60.3→61.5, 38.1→43.5, 45.4→46.1). Pozostałe parametry nie dają spójnej poprawy.
- Mimo to fragmentacja zostaje (AssA ok. 42%). Parametry jej nie rozwiązują.
- Domyślny config nie został zmieniony. Wynik na 5 strojonych sekwencjach trzeba potwierdzić na innych.

### Diagnoza: drużyny i strony (2026-10-09)

Analiza offline w Pythonie na cache biegu `20261008-215833` i zrzutach biegów z tabeli wyżej. Kod pipeline'u bez zmian.
Port (`TrackletTeamClustering`, `TrackletTeamSideLabeling`) jest wierny oryginałowi: KMeans (k=2) na średnich embeddingach ReID torów `player`,
potem klaster o większej średniej x detekcji to „right”. Bramkarze dostają stronę według połowy boiska (to działa: 100% zgodności).

**1. Zamianę stron powoduje domieszka w klastrach, a nie sama reguła średniego x ani kalibracja.**
- SNGS-022: klastry dzielą drużyny w 80% (przy właściwym mapowaniu), ale klaster „left” zawiera 18 torów drużyny prawej.
  To przesuwa jego średnią x i odwraca decyzję przy marginesie 1.2 m. Przy czystych klastrach te same przewidziane pozycje dają poprawny margines +5.5 m.
- Reguła średniego x jest z natury słaba: na GT wszystkich 58 sekwencji valid (idealne klastry, idealne pozycje) różnica średnich drużyn to typowo 2–6 m.
  Myli się w 2 z 58 sekwencji (SNGS-038, SNGS-092), a SNGS-056 wychodzi na granicy 0.0 m. Mimo to jest najlepsza z testowanych (mediana, średnia przycięta,
  głosowanie klatkami, kwantyle, skrajny zawodnik). Ma największy margines względem bootstrapu.
- SNGS-079 (67%) to nie zamiana stron, tylko słaba klasteryzacja: 62% czystości.

**2. „`max_age 30` psuje SNGS-028” to zdegenerowany KMeans, a nie zamiana stron.** Jeden klaster dostaje 1 tor (405 detekcji, z czego 402 to bramkarz z GT
sklasyfikowany jako `player`), drugi 76 torów. W 110 parach bieg×sekwencja zdarza się to 10 razy. Pozostałe przypadki to klaster z 1–3 torów po 1–3 detekcje.
Na tych samych torach sklearnowy KMeans się nie degeneruje (88%). Winna jest inicjalizacja w C#: zwykły k-means++ z `Random(0)` i jeden start.
sklearn używa greedy k-means++.

**3. Embedding ReID słabo koduje drużynę. Kolor koszulki dużo lepiej.** Czystość klastrów (dopasowanie do GT po IoU w obrazie, sklearn KMeans, 5 ziaren):

| cecha toru | 022 | 028 | 038 | 079 | 086 | poprawna drużyna (z regułą strony) | złe strony |
|---|--:|--:|--:|--:|--:|--:|--:|
| embedding ReID, 1 start (jak teraz) | 74% | 94% | 89% | 62% | 75% | 69.2% | 5/25 |
| embedding ReID, 10 startów | 93% | 94% | 91% | 62% | 80% | 84.0% | 0/25 |
| kolor tułowia (mediana Lab, bez trawy), 10 startów | 91% | 94% | 97% | 88% | 93% | **92.7%** | 0/25 |

Kolor tułowia: wycinek 25–75% szerokości i 15–50% wysokości boxa, piksele trawy odrzucone (HSV: H 35–85, S > 60), mediana Lab z co 3. klatki,
mediana po torze, standaryzacja, KMeans k=2.
Sprawdzenie przeciw przeuczeniu na GT (boxy i tory z GT, wszystkie 58 sekwencji valid, 10 startów): czystość średnio 96.1%, mediana 100%,
6 sekwencji poniżej 80%. Strona zła w 3 sekwencjach: 038 i 092 to granica reguły średniego x, 056 przy czystości 83%.

**Symulowany wpływ na GS-HOTA** (podmienione tylko etykiety drużyn w zrzutach baseline'u, reguła strony bez zmian, przeliczone `eval-combine`):

| wariant | pełny | bez nr | pozycja | bez nr per sekwencja (022/028/038/079/086) |
|---|--:|--:|--:|---|
| baseline | 31.9 | 40.5 | 50.3 | 26.4 / 44.1 / 57.8 / 28.1 / 43.4 |
| embedding, 10 startów | 33.6 | 44.8 | 50.3 | 56.2 / 44.3 / 57.8 / 26.9 / 39.7 |
| kolor tułowia, 10 startów | **35.0** | **48.3** | 50.3 | 55.9 / 44.3 / 60.0 / 36.2 / 44.6 |

Same 10 startów pogarsza 079 i 086. Kolor poprawia albo nie zmienia każdej sekwencji.

### Drużyny po kolorze koszulki — wdrożone (2026-10-09)

Nowy moduł `TorsoColor` (poziom detekcji, bez modelu, ok. 0.01 s/klatkę) liczy kolumnę `torso_color` tak samo jak wyżej, ale z każdej klatki.
`TrackletTeamClustering` ma parametr `team.feature` w configu:
- `color` (nowy domyślny): mediana `torso_color` po torze, standaryzacja, KMeans k=2 z 10 startami (najmniejsza inercja);
- `embedding`: zachowanie jak dotąd (średni embedding ReID, 1 start). Daje zrzuty bit w bit identyczne z baseline'em (sprawdzone na 5 sekwencjach).

Reguła strony (`TrackletTeamSideLabeling`) bez zmian. Cache zapisany przed tą zmianą wymaga `RERUN=TorsoColor`; bez tego bieg kończy się błędem o nieaktualnym cache.
Pomiar z cache `20261008-215833`, wynik łączny:

| config | pełny HOTA | bez nr HOTA | bez nr DetA | pozycja HOTA | bez nr per sekwencja (022/028/038/079/086) |
|---|--:|--:|--:|--:|---|
| baseline (`embedding`, `max_age 300`) | 31.9 | 40.5 | 44.5 | 50.3 | 26.4 / 44.1 / 57.8 / 28.1 / 43.4 |
| **`color`, `max_age 300`** (run `20261009-112413`) | **35.0** | **48.3** | **59.9** | 50.3 | 55.9 / 44.3 / 60.0 / 36.3 / 44.6 |
| `embedding`, `max_age 30` (run `20261009-021506`) | 33.2 | 43.5 | 45.0 | 53.2 | 61.5 / 26.3 / 59.1 / 33.3 / 34.8 |
| `color`, `max_age 30` (run `20261009-112725`) | 37.0 | 51.8 | 63.2 | 53.2 | 61.3 / 49.3 / 61.2 / 42.2 / 46.0 |

- Wynik C# pokrywa się z symulacją w Pythonie (35.0 / 48.3 / 50.3).
- Przy kolorze drużyny nie zależą już od drobnych zmian w trackerze: `max_age 30` nie psuje 028 ani 086. Warianty „pełny” i „bez numerów”
  znów nadają się do oceny zmian w trackingu.
- Zostaje: granica reguły średniego x (na GT źle w 038 i 092) i bramkarz rozpoznany jako `player` (028, problem roli).
- Potwierdzenie poza 5 sekwencjami jest na razie tylko na boxach z GT (58 sekwencji, sekcja wyżej). Pełny bieg na innych sekwencjach jeszcze nie był robiony.
