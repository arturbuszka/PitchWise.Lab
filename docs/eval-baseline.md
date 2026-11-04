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

### Diagnoza: fragmentacja torów (2026-10-09)

Analiza offline w Pythonie na cache biegu `20261008-215833` (tory obecnego configu; kolor drużyn nie zmienia trackera). Kod w repo bez zmian.
Predykcje dopasowane do GT w obrazie (IoU ≥ 0.5, Hungarian per klatka). „Zerwanie” to zmiana przewidzianego ID między kolejnymi dopasowaniami tej samej osoby z GT.
Dla każdego zerwania odtworzony stan starego toru (Kalman i EMA embeddingu; zależy tylko od jego własnych detekcji) w klatce, w której osoba dostała nowe ID.
Do pomiaru wpływu na GS-HOTA służy replika StrongSORT.NET w Pythonie. W trybie „jak C#” daje **identyczne `track_id` w 100% detekcji** (sprawdzone na 022 i 028).
Jej ID są wstawiane do zrzutów biegu `20261009-112413` (obecny config) i przeliczane przez `eval-combine`.

**1. Przyczyny zerwań w obecnym trackerze (2484 zerwania na 5 sekwencjach):**

| powód (w klatce nowego ID) | liczba | udział | uwagi |
|---|--:|--:|---|
| bramka Mahalanobisa odrzuciła stary tor | 1964 | 79% | 70% bez żadnej przerwy w detekcji; wygląd pasuje (mediana odl. kos. 0.05 przy progu 0.5) |
| stary tor przegrał w Hungarian (inny tor wziął detekcję) | 446 | 18% | głównie migotanie między dwoma torami tej samej osoby |
| stary tor zajęty inną detekcją (zamiana ID) | 72 | 3% | 78% przy okluzji (IoU z innym GT > 0.3) |
| wygląd > 0.5 | 1 | 0% | |
| tor usunięty po `max_age` | 1 | 0% | |

1499 z 2484 zerwań to powrót do ID, które ta sama osoba już miała, czyli migotanie A→B→A. Suma „liczba ID na osobę” wynosi 1094 przy 538 ID.
Przy zerwaniach bez przerwy bramkę przekracza **wymiar proporcji boxa** (`a = w/h`): mediana jego udziału to 13.8 przy progu 9.49, a pozycja x/y daje 0.8 (błąd ok. 0.1 wysokości boxa).
Zmiana kroku zawodnika między klatkami wystarcza, żeby tor odpadł.

**2. Przyczyna: dwie rozbieżności portu StrongSORT.NET z `bpbreid_strong_sort` z tracklab 1.3.24** (tracker używany przez sn-gamestate):

| | oryginał (`bpbreid_strong_sort/sort/kalman_filter.py`) | StrongSORT.NET |
|---|---|---|
| szum składowej proporcji (init / predict / project) | `2h/20`, `10h/160` / `h/20`, `h/160` / `h/20`, czyli praktycznie bez ograniczenia | `1e-2`, `1e-5` / `1e-2`, `1e-5` / `1e-1` (wartości z pierwotnego StrongSORT; w oryginale są tylko w komentarzu) |
| szum pomiaru w bramce | `gating_distance` woła `project(mean, cov)` bez confidence, czyli pełny szum | `GatingDistance` przekazuje `det.Confidence`, więc szum × (1 − conf); przy conf 0.9 wariancja jest 100× mniejsza |
| liczba predykcji bez pomiaru | `time_since_update < 7` | `<= 7` (drobiazg) |

Drobne różnice, których replika nie zmienia: oryginał robi EMA na nieznormalizowanych cechach, odległość wyglądu liczy jako `sqrt(2-2cos)/2`
(przy progu 0.5 to ta sama bramka co `1-cos ≤ 0.5`), a próg `max_dist` stosuje do sumy `0.995·app + 0.005·maha` (surowa, nie `min(maha/gate, 1)`).

**Wpływ (replika w Pythonie, wynik łączny):**

| tracker | ID | pełny* | bez nr* | pozycja HOTA | pozycja AssA |
|---|--:|--:|--:|--:|--:|
| jak teraz (C#) | 538 | 35.0 | 48.3 | 50.3 | 37.4 |
| tylko bramka bez confidence | 423 | 38.7 | 54.3 | 56.6 | 47.3 |
| tylko szum proporcji jak w oryginale | 377 | 38.9 | 54.4 | 56.4 | 47.2 |
| **obie + `<` (semantyka oryginału)** | 364 | **38.4** | **55.0** | **57.4** | **48.7** |
| semantyka oryginału + `max_age 30` | 426 | 39.1 | 54.9 | 57.2 | 48.1 |
| jak teraz + `max_age 30` (kontrola) | 609 | 36.5 | 51.2 | 53.2 | 41.8 |

\* „pełny” i „bez nr” są przybliżone: rola, drużyna i numer to większość etykiet ze starego biegu w obrębie nowego toru (zastępstwo głosowania na tor).
Kontrola: dla `max_age 30` prawdziwy C# dał 37.0 / 51.8 / 53.2, przybliżenie daje 36.5 / 51.2 / 53.2, czyli lekko zaniża. „Sama pozycja” jest dokładna.

Per sekwencja (pełny / bez nr / pozycja), semantyka oryginału vs teraz:
022 41.0/66.4/70.0 vs 33.2/55.9/57.9; 028 35.0/55.7/61.3 vs 29.6/44.3/50.5; 038 56.4/62.8/64.1 vs 53.5/60.0/60.3;
079 18.6/46.2/47.6 vs 16.8/36.3/38.1; 086 37.7/46.2/47.7 vs 36.5/44.6/45.4. **Poprawa w każdej sekwencji i w każdym wariancie.**
`max_age` przestaje mieć znaczenie, bo zysk z `max_age 30` w obecnym trackerze polegał na ograniczaniu migotania.

**3. Co zostaje po poprawce (semantyka oryginału, 624 zerwania).** Górne granice (wyrocznie z GT), sama pozycja:

| wyrocznia | HOTA | AssA |
|---|--:|--:|
| tracker z semantyką oryginału | 57.4 | 48.7 |
| idealne sklejanie torów (tylko tory rozłączne w czasie) | 62.6 | 57.9 |
| samo rozcięcie zamian ID | 57.6 | 48.7 |
| rozcięcie zamian + sklejanie | 65.7 | 63.4 |
| idealna asocjacja (dopasowane detekcje dostają ID z GT) | 69.7 | 71.4 |
| *dla porównania: idealne sklejanie na obecnym trackerze* | *53.6* | *42.6* |

Na obecnym trackerze nawet idealne sklejanie (53.6) daje mniej niż sama poprawka portu. Przeplatające się tory nie są rozłączne w czasie, więc nie da się ich skleić.

Zysk z idealnego sklejania (+5.3) według przyczyny przerwy między fragmentami (138 linków):

| przyczyna przerwy | linki | zysk HOTA | mediana przerwy | przesunięcie kamery > 200 px |
|---|--:|--:|--:|--:|
| osoba poza kadrem (GT nieoznaczone w ≥ 50% przerwy) | 45 | +2.4 | 122 kl. | 91% |
| okluzja (IoU z innym GT > 0.3) | 44 | +1.1 | 46 kl. | 61% |
| widoczna, bez detekcji lub toru | 45 | +1.2 | 6 kl. | 22% |
| bez przerwy | 4 | +0.1 | 0 | 0% |

Po przerwie dłuższej niż 7 klatek predykcja Kalmana zamarza w obrazie, a kamera w tym czasie się przesuwa (ECC w oryginale też jest wyłączone).
Taki powrót może więc złapać tylko sklejanie offline albo kompensacja ruchu kamery.

**4. Rozróżnialność par do sklejania** (tory rozłączne w czasie, przerwa ≤ 15 s; 5679 par, 311 prawdziwych):
- wygląd (średni embedding toru): AUC 0.82; wśród par z tej samej drużyny i w zasięgu na boisku tylko 0.77. Embedding słabo odróżnia kolegów z drużyny.
- drużyna z `torso_color` (KMeans jak w `TrackletTeamClustering`): zgodna w 83% prawdziwych par i w 33% fałszywych.
  Niezgodności to prawie wyłącznie bardzo krótkie tory (mediana 2 detekcje). Dla torów ≥ 25 detekcji zgodność wynosi 95%.
  Drużyna nadaje się więc na twarde ograniczenie, ale tylko dla torów z wiarygodnym kolorem (np. ≥ 10 detekcji).
- odległość na boisku między końcem i początkiem: mediana 9 m przy medianie przerwy 3.3 s; 90. percentyl prędkości 14.6 m/s (szum kalibracji).

**Prototyp sklejania offline** (na torach z semantyką oryginału): Hungarian koniec toru → początek toru, najwyżej jeden następnik i jeden poprzednik.
Ograniczenia twarde: rozłączność w czasie, przerwa ≤ 10 s, odległość ≤ 3 m + 8 m/s·Δt, ta sama drużyna z koloru (gdy oba tory mają ≥ 10 detekcji),
odległość wyglądu ≤ 0.3. Koszt: `app + 0.3·d/zasięg + 0.1·min(Δt/10 s, 1)`, odrzucenie linku przy koszcie > τ.

| wariant | linki (poprawne / błędne / FP) | pełny* | bez nr* | pozycja HOTA | pozycja AssA |
|---|---|--:|--:|--:|--:|
| bez sklejania | — | 38.4 | 55.0 | 57.4 | 48.7 |
| τ = 0.1 | 102 (63 / 28 / 11) | 40.4 | 58.2 | 60.5 | 54.1 |
| τ = 0.2 | 162 (79 / 65 / 18) | 40.5 | 58.2 | 60.6 | 54.5 |
| bez τ | 183 (80 / 78 / 25) | 40.3 | 58.0 | 60.3 | 54.1 |

Wynik słabo zależy od parametrów: `max_app` 0.2–0.4, `vmax` 6–10, przerwa 5–15 s i próg drużyny 1–25 detekcji dają 60.2–60.6.
To ok. 60% zapasu z wyroczni sklejania. Zysk jest nierówny: pozycja 022 +9.0, 028 +0.2, 038 +1.4, 079 +2.3, 086 +1.2.

### StrongSORT zgodny z tracklab — wdrożone (2026-10-09)

Poprawka z punktu 2 diagnozy wyżej, w StrongSORT.NET (`../StrongSort.NET`). PitchWise korzysta z niego teraz przez `ProjectReference`
zamiast paczki `StrongSort.Net 1.0.0`, więc repo StrongSORT.NET musi leżeć obok PitchWise.Lab (także na Windows).
- `StrongSortOptions.KalmanNoise`: `BpbReid` (domyślnie; szum proporcji `h/20`, `h/160` jak w tracklab) albo `StrongSort` (dawne stałe 1e-2 / 1e-5 / 1e-1);
- `StrongSortOptions.GatingUsesConfidence`: domyślnie `false`, czyli bramka z pełnym szumem pomiaru, jak w tracklab;
- predykcja Kalmana działa, dopóki `time_since_update < MaxKalmanPredictionsWithoutUpdate` (było `<=`);
- config, sekcja `track`: `kalman_noise: bpbreid`, `gating_uses_confidence: false`, `max_kalman_predictions_without_update: 7`.
  Stary tracker odtwarzają `strongsort` / `true` / `8`. Sprawdzone: zrzuty wszystkich 5 sekwencji są identyczne z runem `20261009-112413`.
- testy: StrongSort.Tests 14/14 (4 nowe: powrót po przerwie ze zmienioną proporcją boxa, bramka w trzech trybach, liczba predykcji bez pomiaru), GameState.Tests 48/48.

Sekcja `track` w fingerprincie ma nowe pola, więc cache zapisany wcześniej wymaga `StrongSortModule` w `RERUN`.
Pomiar z cache `20261008-215833` (`RERUN=StrongSortModule,TorsoColor`, run `20261009-135233`), wynik łączny:

| config | pełny HOTA | bez nr HOTA | pozycja HOTA | pozycja AssA | pełny per sekwencja (022/028/038/079/086) | pozycja per sekwencja |
|---|--:|--:|--:|--:|---|---|
| baseline | 31.9 | 40.5 | 50.3 | 37.4 | 16.0 / 29.7 / 53.5 / 16.8 / 35.2 | 57.9 / 50.5 / 60.3 / 38.1 / 45.4 |
| poprzedni domyślny (`color`, tracker jak w 1.0.0) | 35.0 | 48.3 | 50.3 | 37.4 | 33.2 / 29.6 / 53.5 / 16.8 / 36.5 | 57.9 / 50.5 / 60.3 / 38.1 / 45.4 |
| **`color` + tracker zgodny z tracklab** | **39.3** | **55.8** | **57.4** | **48.7** | 45.8 / 32.7 / 56.2 / 19.8 / 38.0 | 70.0 / 61.3 / 64.1 / 47.6 / 47.7 |

- Poprawa w każdej sekwencji i każdym wariancie. „Sama pozycja” zgadza się co do dziesiątej części z repliką w Pythonie (57.4).
  „Pełny” i „bez nr” wyszły wyżej niż przybliżenie (38.4 / 55.0), bo prawdziwe głosowanie na tor działa lepiej.
- Zmiana przywraca zachowanie oryginału, a nie stroi parametrów, więc ryzyko przeuczenia pod 5 sekwencji jest małe.
  Pełnego biegu poza stałą piątką nadal nie było.

### Sklejanie torów (`TrackletMerging`) — wdrożone (2026-10-09)

Nowy moduł poziomu wideo [`TrackletMerging`](../src/PitchWise.GameState/Track/TrackletMerging.cs) to port prototypu z punktu 4 diagnozy fragmentacji.
Stoi za `NbjwCalib`, bo potrzebuje `bbox_pitch`, i przed `MajorityVoteTracklet`. Czyta `track_id`, `bbox_pitch`, `embeddings`, `torso_color` i `role_detection`,
a nadpisuje `track_id`: cały łańcuch fragmentów dostaje ID pierwszego toru.
- **Config:** sekcja `tracklet_merging` z `enabled: false` i parametrami prototypu. Ma własny fingerprint w `PipelineConfig`.
  Gdy moduł jest włączony, pipeline dokłada `TorsoColor` także przy `team.feature: embedding`.
- **Ograniczenia i koszt** jak w prototypie, z jednym uzupełnieniem: jeśli jest tylko jeden wiarygodny tor zawodnika, nie da się go sklasyfikować do drużyny.
  Wtedy i tak nie skleja się go z bramkarzem ani sędzią (w prototypie ten przypadek nie występuje).
- **Odstęp w sekundach** liczony z `frame_rate` wideo; `BuildPipeline` dostał parametr `fps`, a jeśli go brakuje, przyjmowane jest 25.
- **Zmiany w trybie cache**, bo to pierwszy moduł, który nadpisuje kolumnę, którą sam czyta:
  - `OfflineTrackingEngine.ClearOutputs` nie czyści już takiej kolumny przed ponownym uruchomieniem modułu.
    Wcześniej kasował `track_id` i moduł widział 0 torów.
  - `--rerun TrackletMerging` dokłada wcześniejszego producenta kolumny (`StrongSortModule`). Bez tego moduł skleiłby ID już sklejone w cache.
- **Testy:** 12 nowych (`TrackletMergingTests`), GameState.Tests 60/60.
  Przy wyłączonym module zrzuty 5 sekwencji są bit w bit identyczne z runem `20261009-135233` (run kontrolny `20261009-170858`).

**Zgodność z prototypem:** replika trackera w Pythonie daje te same `track_id` co C# w 100% detekcji, więc obie strony sklejają te same tory.
Po sklejeniu podział ocenianych detekcji na tory jest identyczny we wszystkich 5 sekwencjach (0 rozbieżnych detekcji).
Liczba linków też się zgadza: 102 (022: 23, 028: 13, 038: 11, 079: 24, 086: 31), z czego 63 poprawne, 28 błędnych i 11 z fałszywymi detekcjami.

Pomiar z cache `20261008-215833` (`CONFIG` = kopia domyślnego configu z `tracklet_merging.enabled: true`, `RERUN=StrongSortModule,TorsoColor`, run `20261009-170459`), wynik łączny:

| config | pełny HOTA | bez nr HOTA | pozycja HOTA | pozycja AssA | pełny per sekwencja (022/028/038/079/086) | bez nr per sekwencja | pozycja per sekwencja |
|---|--:|--:|--:|--:|---|---|---|
| baseline | 31.9 | 40.5 | 50.3 | 37.4 | 16.0 / 29.7 / 53.5 / 16.8 / 35.2 | 26.4 / 44.1 / 57.8 / 28.1 / 43.4 | 57.9 / 50.5 / 60.3 / 38.1 / 45.4 |
| obecny domyślny (`color` + tracker zgodny z tracklab) | 39.3 | 55.8 | 57.4 | 48.7 | 45.8 / 32.7 / 56.2 / 19.8 / 38.0 | 69.5 / 56.1 / 62.9 / 46.4 / 46.9 | 70.0 / 61.3 / 64.1 / 47.6 / 47.7 |
| **+ `tracklet_merging`** | **40.4** | **58.8** | **60.5** | **54.1** | 49.0 / 32.7 / 56.7 / 19.4 / 39.4 | 77.4 / 58.8 / 63.6 / 48.3 / 48.9 | 77.9 / 64.0 / 64.8 / 49.5 / 49.9 |

- **Sama pozycja** zgadza się z prototypem co do dziesiątej części, łącznie i w każdej sekwencji (60.5, AssA 54.1).
  Liczba ID spada z 364 do 262 (o liczbę linków).
- **Wariant pełny** wyszedł 40.4, tak jak przybliżenie w prototypie. Wariant „bez numerów” wyszedł 58.8 wobec 58.2 z przybliżenia, bo prawdziwe głosowanie na tor działa lepiej.
- **„Sama pozycja” i „bez numerów”** rosną w każdej sekwencji. Wariant pełny rośnie w 3 sekwencjach, w 028 stoi w miejscu, a w 079 spada o 0.4 (19.8 → 19.4).
  W 079 DetA wariantu pełnego spada 11.4 → 10.3. Prawdopodobna przyczyna, jeszcze niesprawdzona: błędny link łączy dwie osoby, a głosowanie daje wtedy jeden numer na oba fragmenty.
- **Korekta do opisu prototypu wyżej:** rozkład zysku „022 +9.0, 028 +0.2, …” nie pochodzi z wariantu τ = 0.1. Prototyp przeliczony ponownie przy τ = 0.1
  daje te same liczby co C#: +7.9 / +2.7 / +0.7 / +1.9 / +2.2 (pozycja). Zysk jest więc mniej skupiony w 022, niż podawał ten opis.
- **Parametry dobrane na tych samych 5 sekwencjach.** Moduł zostaje wyłączony, dopóki wynik nie potwierdzi się na innych sekwencjach.

### Potwierdzenie na sekwencjach kontrolnych (2026-10-09)

10 sekwencji valid spoza stałej piątki, wylosowanych raz ze stałym seedem: [`eval-sequences-holdout.txt`](eval-sequences-holdout.txt).
Nic nie było na nich strojone. Skrypty przyjmują teraz inną listę sekwencji: `SEQS_FILE=<plik>` (`run_eval.sh`) albo `-SeqsFile` (`run_eval.ps1`).
- Pełny bieg obecnego domyślnego configu z wyłączonym OCR (`jersey.enabled: false`): run `20261009-211433`, ok. 1 h 45 min, wszystkie modele na CUDA.
- Pozostałe warianty z jego cache: baseline `20261009-225918`, poprzedni domyślny `20261009-230348`, sklejanie `20261009-230818`.
- Bez OCR wariant „pełny” jest nieważny, więc go pomijam.

Wynik łączny (10 sekwencji, 7138 klatek):

| config | bez nr HOTA | pozycja HOTA | pozycja AssA |
|---|--:|--:|--:|
| baseline (stary tracker, `embedding`) | 23.4 | 47.6 | 36.5 |
| poprzedni domyślny (stary tracker, `color`) | 27.3 | 47.6 | 36.5 |
| obecny domyślny (`color` + tracker zgodny z tracklab) | 31.5 | 54.1 | 47.1 |
| **+ `tracklet_merging`** | **34.0** | **57.7** | **53.6** |

Sama pozycja per sekwencja (023 / 031 / 034 / 041 / 044 / 048 / 050 / 056 / 082 / 092):

| config | HOTA |
|---|---|
| baseline | 34.6 / 40.5 / 42.7 / 68.0 / 44.1 / 44.3 / 36.1 / 74.3 / 45.2 / 46.9 |
| obecny domyślny | 38.2 / 47.5 / 51.4 / 72.6 / 51.7 / 52.1 / 41.4 / 80.6 / 54.6 / 54.2 |
| + `tracklet_merging` | 40.8 / 53.9 / 55.5 / 77.1 / 54.3 / 54.4 / 42.3 / 80.8 / 60.2 / 61.9 |

Linki: 236 w 10 sekwencjach (17 / 33 / 21 / 9 / 32 / 37 / 7 / 4 / 34 / 42).

- **Tracker zgodny z tracklab:** +6.5 w „samej pozycji” (47.6 → 54.1) i +10.6 AssA, poprawa w każdej z 10 sekwencji. Na stałej piątce było +7.1.
- **Sklejanie:** kolejne +3.6 (54.1 → 57.7) i +6.5 AssA. Też poprawa w każdej sekwencji, od +0.2 (056, 4 linki) do +7.7 (092). Na stałej piątce było +3.1.
  „Bez numerów” rośnie w 9 z 10 sekwencji, w 056 spada o 0.1.
- **Kolor drużyn** (poprzedni domyślny vs baseline, ten sam tracker): +3.9 w „bez numerów”, ale nierówno.
  Poprawa w 023, 031, 034, 044, 056 i 082. Wyraźne spadki w 041 (31.2 → 15.7) i 092 (27.1 → 18.4); 092 to jedna z dwóch sekwencji, w których reguła strony myli się już na GT.
- **Nowy problem: „bez numerów” na tych sekwencjach jest bardzo niski** (31.5 wobec 55.8 na piątce).
  DetA „bez numerów” wynosi ok. 25%, przy DetA „samej pozycji” 62%. Na piątce było 63% wobec 68%.
  Na przykład 041 ma 72.6 w pozycji i 17.8 bez numerów. Przyczyny (zamiana stron, rola czy klasteryzacja) jeszcze nie sprawdziłem.

**Decyzja (2026-10-10):** `tracklet_merging.enabled: true` w `configs/soccernet.yaml`, bo wynik potwierdził się na sekwencjach kontrolnych.
Tabela między markerami to dalej baseline sprzed zmian. Szybki bieg z cache `20261008-215833` wymaga teraz tego samego `RERUN=StrongSortModule,TorsoColor`.
`TrackletMerging` przelicza się sam, bo czyta `track_id`.

### Diagnoza: niski wariant „bez numerów” na sekwencjach kontrolnych (2026-10-10)

Analiza offline zrzutów obecnego configu: sekwencje kontrolne z runu `20261009-230818`, stała piątka z runu `20261009-170459`.
Predykcje dopasowane do GT na boisku: Hungarian w każdej klatce, do 2 m.

**1. Główna przyczyna to rola z PRTReid: cała jedna drużyna dostaje `referee`.**
Zgodność ról na dopasowanych detekcjach:

| sekwencja | 023 | 031 | 034 | 041 | 044 | 048 | 050 | 056 | 082 | 092 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| rola zgodna z GT | 97% | 94% | 99% | **48%** | **45%** | **47%** | **43%** | **53%** | 100% | 100% |

- W 041, 044, 048, 050 i 056 zawodnicy jednej drużyny z GT są prawie w całości `referee`, w 048 i 050 częściowo też `goalkeeper`.
  Na przykład w 041 drużyna prawa: 18 detekcji `player`, 3887 `referee`, 137 `goalkeeper`; drużyna lewa jest prawie cała `player`.
- Prawdziwi sędziowie dostają w tych sekwencjach rolę `player`.
- Druga drużyna zostaje wtedy jedyną drużyną „zawodników”, więc KMeans dzieli ją na dwie. Stąd zgodność drużyny 18–73% w tych sekwencjach.
- Na stałej piątce tego nie ma: zgodność ról 91–100%. Gorzej jest tylko w 028, gdzie bramkarz dostaje `player`, a część sędziów też `player`.
- Bramkarz z GT dostaje rolę `player` w 7 z 10 sekwencji kontrolnych, czyli to samo zjawisko co w 028.

**To nie błąd portu.** Na 305 wycinkach z GT sekwencji 041 model w PyTorch daje dokładnie te same role co nasz ONNX (305/305).
Model w PyTorch był uruchomiony jak w sn-gamestate: `FeatureExtractor` z prtreid@30617a7, wycinki RGB z tracklab 1.3.24, wagi `prtreid-soccernet-baseline.pth.tar`.
Model tak samo oznacza drużynę prawą jako `referee` (110 ze 124 wycinków). Preprocessing w C# (RGB, resize liniowy, normalizacja ImageNet) też odpowiada oryginałowi.
Głowa ról nie uogólnia się więc na stroje spoza tego, na czym była trenowana.

**2. Druga przyczyna: zamiana stron w 092.** Drużyny są podzielone poprawnie (czystość 96%), ale strony są odwrócone (zgodność 3.9%).
To znany przypadek, w którym reguła średniego x myli się już na GT.

**Wyrocznie** (etykieta toru = większość z dopasowanego GT; „bez numerów”, wynik łączny):

| wariant | kontrolne (10) | stała piątka |
|---|--:|--:|
| obecny config | 34.0 | 58.8 |
| poprawiona tylko zamiana stron (cała sekwencja) | 41.3 | 58.8 |
| idealna drużyna toru, rola z modelu | 44.4 | 59.0 |
| idealna rola toru, drużyna z pipeline'u | 37.2 | 58.8 |
| idealna rola i drużyna toru | **57.3** | 59.8 |
| *sama pozycja (górna granica)* | *57.7* | *60.5* |

Na sekwencjach kontrolnych role i drużyny kosztują ok. 23 pkt, na piątce ok. 1.
Sama poprawa drużyn daje tylko część zysku, bo toru z rolą `referee` nie obejmuje klasteryzacja drużyn. Pełny zysk wymaga poprawienia roli i drużyny razem.

### Korekta ról po kolorze koszulki (`TrackletRoleCorrection`) (2026-10-10)

Analiza offline w Pythonie, potem moduł w C#. Role z naszego ONNX (CPU) i kolor tułowia (jak w `TorsoColor`) policzone na boxach z GT
co 10. klatkę we wszystkich 58 sekwencjach valid. Rola toru GT to głosowanie ważone pewnością, jak w `MajorityVoteTracklet`.

**1. Skala problemu.** Valid to tylko 3 mecze: game 2 (021–038), game 3 (039–059) i game 5 (078–096).
- **Cała drużyna jako `referee`:** 21 z 58 sekwencji, czyli **wszystkie sekwencje meczu 3** i żadna z dwóch pozostałych meczów.
  - Dotyczy drużyny w żółtych koszulkach (Lab ok. 185/116/180). W 039–048 to drużyna prawa, w 049–059 lewa, czyli ta sama drużyna po zmianie stron.
  - Tej drużynie model daje `referee` w 59–98% detekcji, a `goalkeeper` w 1–37%.
  - Prawdziwi sędziowie dostają tam `player` w 32–89% detekcji.
  - Średnio w każdej klatce widać 7–9 torów z głosem `referee`, w porywach 11.
    W meczach 2 i 5 średnio najwyżej 1.7, a maksimum to 3. Z GT wychodzi średnio najwyżej 1.8 sędziego na klatkę.
- **Bramkarz z GT ma w torze głos `player`:** 28 z 58 sekwencji. To 7 sekwencji meczu 2, prawie cały mecz 3 i 3 sekwencje meczu 5.
- Problem z sędziami to więc jeden strój w jednym meczu, a nie zjawisko widoczne w wielu meczach.
  Stała piątka nie ma żadnej sekwencji z meczu 3. Z sekwencji kontrolnych należą do niego 041, 044, 048, 050 i 056.

**2. Reguła** (projekt i wybór parametrów na torach i boxach z GT wszystkich 58 sekwencji).
Kolor toru to mediana Lab, a L ma wagę 0.25. Kanał a/b rozdziela drużyny czysto, natomiast w meczu 3 cień zmienia L w jednej drużynie od 83 do 204.
Przy wadze L 0.5 albo przy standaryzacji kanałów prototypy drużyn rozjeżdżają się w 048 i 056.
1. **Zatłoczona grupa to drużyna.** Wyzwalacz: średnio więcej niż 3 detekcje `referee` albo więcej niż 2 detekcje `goalkeeper` na klatkę.
   - Prototypy drużyn to mediana koloru zatłoczonej grupy i mediana koloru grupy `player`.
   - Tor bliżej niż 20 od któregoś prototypu dostaje rolę `player`.
   - Tor z głosem `player`, który jest dalej od obu prototypów, staje się „outsiderem”.
2. **Bramkarz.** Prototypy drużyn to mediany klastrów KMeans (k=2) z torów `player`.
   - Tor `player` albo outsider, który jest dalej niż 15 od obu prototypów i ma średnie |x| > 38 m, dostaje rolę `goalkeeper`.
   - Na każdą połowę najwyżej jeden taki tor: ten najdalszy w kolorze.
   - Pozostali outsiderzy dostają rolę `referee`.
   - Na GT bramkarze z głosem `player` mają |x| ≥ 41 m i odległość ≥ 18. Zawodnicy daleko od kolorów drużyn mają |x| ≤ 35 m.

Zgodność z GT (rola, drużyna z KMeans i reguły strony, oraz obie naraz; średnia po sekwencjach, liczona na wszystkich detekcjach GT):

| mecz | rola | drużyna | rola i drużyna | sekwencje gorsze niż bez reguły |
|---|---|---|---|---|
| game 2 (18) | 98.5% → 99.8% | 92.9% → 94.0% | 93.3% → 94.3% | 036 (−0.4) |
| game 3 (21) | 45.6% → 80.9% | 31.9% → 83.6% | 25.3% → **80.0%** | brak (każda co najmniej +30) |
| game 5 (19) | 99.7% → 99.9% | 89.5% → 94.6% | 90.4% → 95.0% | brak |
| wszystkie 58 | 79.7% → 93.0% | 69.7% → 90.4% | 67.7% → **89.4%** | |

- W meczach 2 i 5 wyzwalacz nie odpala ani razu. Zmiany pochodzą tylko z reguły bramkarza.
  Poprawia ona też stronę w 084 (13% → 100%), bo bramkarz liczony jako zawodnik przesuwał średnie x klastra.
- **Wrażliwość:** parametry dobrane w siatce (waga L 0 / 0.1 / 0.25 / 0.5, R 10–30, D 10–20). W okolicy wybranego punktu wynik dla meczu 3 skacze.
  Na przykład R 25 daje 059: 17%, a D 20 daje 042: 4%. To odwrócone strony: reguła średniego x jest czuła na to, kto trafi do drużyny.
  Sam krok 1 bez kroku bramkarza odwraca strony w 042 i 058.
- W 045, 048 i 058 zostaje 58–74%: część zawodników w głębokim cieniu wypada poza promień i dostaje `referee`.
  Ciemnych sędziów (L ok. 20–70) nie da się kolorem odróżnić od białej drużyny w cieniu.

**3. Symulacja** na zrzutach obecnego configu. Tory z pipeline'u, czyli z cache nowego biegu z tymi samymi wynikami co `20261009-230818` i `20261009-170459`.
Minimum 10 detekcji z kolorem. Drużyny i strony przeliczone przez pythonowy odpowiednik `TrackletTeamClustering` i `TrackletTeamSideLabeling`,
który na tych torach daje te same etykiety co C# (100%; w 050 99.98%). Wynik: piątka 41.0 / 59.4 / 60.5, kontrolne 47.2 / 57.7.

**4. Moduł** [`TrackletRoleCorrection`](../src/PitchWise.GameState/Team/TrackletRoleCorrection.cs) (poziom wideo, ok. 0.1 s na sekwencję)
stoi za `MajorityVoteTracklet` i przed `TrackletTeamClustering`.
- Czyta `image_id`, `track_id`, `role`, `torso_color` i `bbox_pitch`, a nadpisuje `role`.
  `--rerun TrackletRoleCorrection` sam dokłada więc `MajorityVoteTracklet`.
- **Config:** sekcja `role_correction` z `enabled: false` i parametrami jak wyżej, z fingerprintem w `PipelineConfig`.
  Wymaga reid i kalibracji. Gdy moduł jest włączony, pipeline dokłada `TorsoColor`.
- **Testy:** 8 nowych (`TrackletRoleCorrectionTests`), GameState.Tests 68/68.

Pomiar z cache:
- piątka: `20261008-215833`, `RERUN=StrongSortModule,TorsoColor`, run `20261010-101602`;
- kontrolne: `20261009-211433`, `RERUN=TrackletMerging`, OCR wyłączony, run `20261010-101828`.

| config | piątka pełny | piątka bez nr | piątka pozycja | kontrolne bez nr | kontrolne pozycja |
|---|--:|--:|--:|--:|--:|
| baseline | 31.9 | 40.5 | 50.3 | 23.4 | 47.6 |
| obecny domyślny | 40.4 | 58.8 | 60.5 | 34.0 | 57.7 |
| **+ `role_correction`** | **41.0** | **59.4** | 60.5 | **47.2** | 57.7 |

Bez numerów per sekwencja (obecny domyślny → z korektą):

| sekwencja | 022 | 028 | 038 | 079 | 086 |
|---|---|---|---|---|---|
| piątka | 77.4 → 77.4 | 58.8 → **63.3** | 63.6 → 63.6 | 48.3 → 48.3 | 48.9 → 48.8 |

| sekwencja | 023 | 031 | 034 | 041 | 044 | 048 | 050 | 056 | 082 | 092 |
|---|---|---|---|---|---|---|---|---|---|---|
| kontrolne | 39.5 → 40.9 | 51.0 → 51.0 | 55.0 → 55.0 | 17.8 → **71.4** | 27.9 → **43.8** | 20.0 → **39.3** | 21.4 → **38.1** | 27.7 → **57.7** | 58.7 → 58.7 | 19.4 → 19.4 |

- **C# odtwarza symulację:** wynik łączny i każda sekwencja co do dziesiątej części.
  Rola i drużyna są identyczne w 145 tys. predykcji z wyjątkiem 2 (031).
- **Wyzwalacz odpala dokładnie w 5 sekwencjach meczu 3** (zatłoczona grupa `referee`; zmienia 6–22 torów).
  W pozostałych 10 sekwencjach zmienia najwyżej 1 tor, zawsze na `goalkeeper`.
  W 028, 023 i 031 to bramkarz z GT. W 086 to fałszywy bramkarz: tor sklejony z zawodników obu drużyn, z kolorem pomiędzy nimi, przy bramce (−0.1).
- Wariant pełny na piątce: 40.4 → 41.0 (028: 32.7 → 37.3).
- Z górnej granicy dla sekwencji kontrolnych (57.3 przy idealnej roli i drużynie) zostaje ok. 10 pkt.
  Na to składają się: 092 (odwrócone strony, nietknięte), cień w 044/048/050 i sędziowie oznaczeni jako zawodnicy.
- **Zastrzeżenie:** część „drużyna jako sędziowie” jest zaprojektowana i dostrojona na GT meczu 3, a z tego meczu pochodzi 5 sekwencji kontrolnych.
  Ich zysk (+13 pkt) nie jest więc niezależnym potwierdzeniem, tylko pokazuje, że reguła z GT przenosi się na tory z pipeline'u.
  Niezależnym sprawdzianem byłby inny mecz z tym problemem. Lokalnie jest tylko split valid, więc wymagałoby to danych train lub test.
  Reguła bramkarza działa w trzech meczach.
- Moduł zostaje wyłączony (`enabled: false`) do decyzji.

**Decyzja (2026-10-10):** `role_correction.enabled: true` w `configs/soccernet.yaml`.
- Wyzwalacz części „drużyna jako sędziowie” w meczach bez problemu nie odpala.
- Tam, gdzie odpala, wynik bez korekty jest bardzo zły.
- Reguła bramkarza działa w trzech meczach.
- Zastrzeżenie o meczu 3 zostaje: część „drużyna jako sędziowie” trzeba sprawdzić na innym meczu z tym problemem (split train lub test).

Sprawdzone: domyślny config z cache daje zrzuty bit w bit identyczne z runami `20261010-101602` (piątka) i `20261010-101828` (kontrolne).
Runy kontrolne: `20261010-154834` i `20261010-155057`.
Obecny domyślny config daje więc:
- stała piątka: **41.0 / 59.4 / 60.5** (pełny / bez nr / pozycja);
- sekwencje kontrolne: **47.2 / 57.7** (bez nr / pozycja).

### Strona drużyn: 092 to błąd etykiet GT, nie reguły (2026-10-10)

Analiza na GT wszystkich 58 sekwencji valid (tory, drużyny i pozycje z GT) i na zrzutach runów `20261010-101602` / `20261010-101828`. Kod bez zmian.

**1. W 092 etykiety drużyn zawodników w GT są zamienione.**
- Drużyna w białych koszulkach (Lab ok. 190/125/129) ma etykietę `right` w każdej sekwencji drugiej połowy meczu 5 (086–096), poza 092.
  - 092 zaczyna się w 2-33:37, czyli między 091 (29:10) i 093 (34:20). W obu tych sekwencjach biali są `right`.
  - W trakcie połowy strony się nie zmieniają.
- W 092 wszystkie sygnały z ustawienia wskazują przeciwnie do etykiet zawodników:
  - średnie x: −3.8 m;
  - głos „drugi najbardziej wysunięty”: −0.74;
  - głos „drugi najgłębszy”: −0.68;
  - bliskość bramkarzy: −1.00 / −0.31.
- Na klatkach 20 i 700 niebiescy (w GT `right`) bronią lewej bramki, przy bramkarzu oznaczonym `left`.
- Pipeline przypisuje więc strony w 092 poprawnie, a GS-HOTA liczy to jako błąd.
  Wyrocznia „poprawione strony” (+7.3 na kontrolnych, sekcja wyżej) mierzyła ten błąd etykiet, a nie zapas do odzyskania.
  Metryki ani GT nie zmieniam. Sekwencję 092 trzeba czytać z tym zastrzeżeniem.
- W meczu 2 etykiety są spójne: biali to `right` w pierwszej połowie (021–028) i `left` w drugiej (029–038). Także w 038.

**2. 038 to rzeczywista granica reguły średniego x.** Klip toczy się pod bramką R: stały fragment, obie drużyny w polu karnym.
Margines na GT wynosi −1.0 m. Na torach z pipeline'u 038 akurat wychodzi dobrze (zgodność drużyny 92%).

**3. Sygnały sprawdzone na GT** (poza wariantami z sekcji „Diagnoza: drużyny i strony”):

| sygnał | błędne sekwencje na GT (bez 092) |
|---|---|
| średnie x klastra (obecna reguła) | 038; 056 na granicy (0.0 m) |
| głos klatkami: drugi najbardziej wysunięty zawodnik | 030, 038, 054, 086 (0.00) |
| głos klatkami: drugi najgłębszy zawodnik | 037, 038, 056 (−0.01) |
| bliskość bramkarza: 3 najbliżsi z każdej drużyny, strona bramkarza wg połowy | 030, 093, niejednoznaczne 043 i 058 (68 z 75 bramkarzy ma bliżej swoich) |

Połączenie „przy marginesie < 1.5 m decyduje bramkarz” naprawia 038, ale psuje 030 (margines 0.5 m, głos bramkarza −0.45).
Na GT bilans wynosi zero, więc to byłoby strojenie pod dwa przypadki.

**4. Stan w pipeline (z `role_correction`):** zgodność drużyny zawodników z GT 90–99% we wszystkich 15 sekwencjach poza 092.
Bez korekty ról odwrócone wyglądały też 041 (18%), 048 (40%) i 056 (37%). To był skutek złych ról, a nie reguły strony.

**Wniosek:** reguły lepszej od średniego x nie znalazłem. Na obecnych 15 sekwencjach jedyna zła strona to błąd GT, więc modułu nie dodaję.
Punkt 2 kolejki można zamknąć. Wraca, jeśli na nowych danych pojawią się zamiany stron przy poprawnych etykietach.

### Kompensacja ruchu kamery w trackerze (`track.camera_motion`) (2026-10-10)

Problem z diagnozy fragmentacji: po przerwie dłuższej niż 7 klatek predykcja Kalmana staje w miejscu w obrazie, a kamera się przesuwa.
Oryginał ma na to ECC (`bpbreid_strong_sort/ecc.py`), ale w sn-gamestate jest ono wyłączone (`ecc: False`).

**1. Jakość estymacji ruchu kamery** (analiza na GT, sekwencje 022, 028, 030, 045, 079, 090).
- Wzorzec: homografia obraz→boisko w każdej klatce, dopasowana do stóp z boxów GT i ich `bbox_pitch_raw`.
  Prawdziwy ruch płaszczyzny boiska w obrazie między t i t+g to `inv(H[t+g]) · H[t]`.
- Błąd zmierzony na stopach GT z klatki t, w wysokościach boxa (mediana):

| przerwa | bez CMC | ECC Euclidean, skala 0.1 (tracklab) | ECC affine | przepływ optyczny (LK + RANSAC, podobieństwo, maska boxów) |
|---|---|---|---|---|
| 10 klatek | 0.19–1.15 | 0.12–0.18 | 0.11–0.26 | **0.05–0.14** |
| 50 klatek | 0.59–6.24 | 0.42–0.70 | 0.37–1.32 | **0.13–0.54** |

- Przepływ optyczny jest najdokładniejszy, zwłaszcza przy dłuższych przerwach. Euclidean z tracklab nie ma skali, więc nie łapie zoomu.

**2. Implementacja** (domyślnie wyłączona, zachowanie bez zmian):
- StrongSort.NET: `StrongSortTracker.Update(observations, cameraMotion)` przyjmuje opcjonalną macierz 3×3 (piksele poprzedniej klatki → bieżąca).
  Po predykcji przesuwa box każdego toru jak `Track.camera_update` w tracklab: przekształca dwa rogi, prędkości i kowariancji nie rusza.
  Testy: StrongSort.Tests 17/17 (3 nowe).
- PitchWise: [`CameraMotion`](../src/PitchWise.GameState/Track/CameraMotion.cs) liczy macierz z obrazów w `StrongSortModule`; config `track.camera_motion`:
  - `none` (domyślnie);
  - `ecc`: jak `ecc.py` z tracklab (Euclidean, skala 0.1, 100 iteracji, eps 1e-5, `get_matrix`);
  - `flow`: rogi poza boxami osób z poprzedniej klatki, LK, `estimateAffinePartial2D` z RANSAC, skala 0.5.
- Testy: GameState.Tests 71/71 (3 nowe, `CameraMotionTests`).
- Przy `none` zrzuty piątki są bit w bit identyczne z runem `20261010-154834` (run `20261010-212138`).
- Koszt: `StrongSortModule` 0.01 → 0.03–0.04 s/klatkę, głównie wczytanie obrazów.

**3. Pomiar z cache** (obecny domyślny config + `camera_motion`; piątka: `20261008-215833`, `RERUN=StrongSortModule,TorsoColor`; kontrolne: `20261009-211433`, `RERUN=TrackletMerging`, bez OCR):

| config | piątka pełny | piątka bez nr | piątka pozycja | piątka pozycja AssA | kontrolne bez nr | kontrolne pozycja | kontrolne pozycja AssA |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 31.9 | 40.5 | 50.3 | 37.4 | 23.4 | 47.6 | 36.5 |
| obecny domyślny (`none`) | 41.0 | 59.4 | 60.5 | 54.1 | 47.2 | 57.7 | 53.6 |
| `ecc` (runy `20261010-212406`, `20261010-213125`) | 42.9 | 61.8 | 62.6 | 57.9 | 44.4 | 57.6 | 53.3 |
| `flow` (runy `20261010-212755`, `20261010-213748`) | 42.6 | 61.2 | 62.2 | 57.3 | 48.4 | 57.7 | 53.5 |

Pozycja HOTA per sekwencja (`none` / `ecc` / `flow`):

| piątka | 022 | 028 | 038 | 079 | 086 |
|---|---|---|---|---|---|
| | 77.9 / 78.0 / 77.3 | 64.0 / **61.4** / **62.4** | 64.8 / 66.8 / 67.0 | 49.5 / 52.3 / 51.3 | 49.9 / **56.5** / **55.6** |

| kontrolne | 023 | 031 | 034 | 041 | 044 | 048 | 050 | 056 | 082 | 092 |
|---|---|---|---|---|---|---|---|---|---|---|
| | 40.8 / 39.6 / 40.5 | 53.9 / **51.1** / 53.9 | 55.5 / **52.5** / **51.4** | 77.1 / 78.5 / 77.3 | 54.3 / 54.6 / 53.7 | 54.4 / 56.6 / 56.2 | 42.3 / 43.1 / 43.2 | 80.8 / 79.0 / 80.2 | 60.2 / 59.1 / 60.0 | 61.9 / 63.7 / 63.2 |

Bez numerów, kontrolne: 023 40.9 / 39.2 / 40.6, 031 51.0 / 48.3 / 52.8, 034 55.0 / 51.0 / 50.0, 041 71.4 / 73.0 / 71.8, 044 43.8 / 46.2 / 44.9,
048 39.3 / 42.3 / 43.2, 050 38.1 / 37.5 / 40.6, 056 57.7 / **26.5** / 63.1, 082 58.7 / 58.2 / 59.7, 092 19.4 / 18.8 / 18.7.

- **Piątka:** +2.1 (`ecc`) i +1.7 (`flow`) w „samej pozycji”. Zysk pochodzi z 038, 079 i 086; 028 traci 1.6–2.6.
- **Kontrolne:** w „samej pozycji” bez zmian (57.7 → 57.6 / 57.7). Poprawa w 041, 048, 050 i 092, spadki w 023, 031, 034 i 082 (034 traci 3–4 pkt).
- **Skok w „bez numerów” to w dużej mierze strona drużyn, a nie tracking.** 056 przy `ecc` ma odwrócone strony (zgodność drużyny zawodników 29% wobec 99%),
  przy pozycji prawie bez zmian. 056 to znana granica reguły średniego x (0.0 m na GT).
- Liczba ID i rozbicie osób GT na ID spadają w prawie każdej sekwencji. Mimo to AssA w 028, 031, 034 i 082 spada.
  Przykład z 028 (tory bez sklejania): zawodnik zasłonięty przez ok. 47 klatek biegnie razem z kamerą. Kamera przesuwa obraz o ok. −13 px/klatkę, a zawodnik w obrazie tylko o −4 px/klatkę.
  Bez CMC zamrożony box czeka bliżej niego i tor wraca. Z CMC box odjeżdża z kamerą i zawodnik dostaje nowe ID.

**4. Dlaczego zysk jest niestabilny: która hipoteza ruchu trafia lepiej** (GT 18 sekwencji z trzech meczów, prawdziwy ruch kamery z homografii GT).
Błąd przewidzianego położenia stóp po przerwie, w wysokościach boxa: mediana (udział < 0.5 boxa). Prędkość to średnia z 5 klatek przed przerwą.

| przerwa | stoi w obrazie | 7 klatek prędkości w obrazie (tracklab bez ECC, obecnie) | stoi na boisku | 7 klatek prędkości na boisku (tracklab z ECC) | stała prędkość na boisku |
|---|---|---|---|---|---|
| 5 | 0.20 (80%) | 0.11 (91%) | 0.18 (86%) | 0.11 (89%) | 0.11 (89%) |
| 10 | 0.39 (59%) | **0.17 (84%)** | 0.32 (67%) | 0.20 (82%) | 0.20 (80%) |
| 25 | 0.92 (31%) | 0.71 (38%) | 0.73 (37%) | 0.59 (44%) | **0.49 (50%)** |
| 50 | 1.71 (18%) | 1.52 (20%) | 1.39 (23%) | 1.26 (24%) | **1.15 (24%)** |
| 100 | 2.88 (9%) | 2.78 (9%) | 2.63 (13%) | **2.53 (13%)** | 2.93 (8%) |

- Kamera transmisyjna podąża za akcją, więc część zawodników porusza się razem z nią. Dla nich „stoi w obrazie” jest lepszym założeniem niż „stoi na boisku”.
  Przy przerwie do ok. 10 klatek CMC w wersji z tracklab trafia więc nieco gorzej niż obecny tracker. Lepiej trafia dopiero przy dłuższych przerwach,
  ale wtedy błąd obu wersji (≥ 0.6 boxa) jest już zwykle poza bramką ruchu.
  W części sekwencji (np. 021, 030, 081, 094) CMC przegrywa nawet przy przerwie 50 klatek.
- Lepsza estymacja ruchu kamery (`flow`) tego nie zmienia. Ogranicza ją założenie o ruchu osoby, a nie dokładność macierzy.

**Wniosek:** CMC jak w tracklab daje zysk na piątce (+1.7–2.1), ale na sekwencjach kontrolnych wynik stoi w miejscu przy dużym rozrzucie per sekwencja.
Zostaje w kodzie z `camera_motion: none`. Możliwe dalsze kroki (niesprawdzone, każdy rusza wyniki):
- CMC tylko dla torów bez aktualizacji dłużej niż 7 klatek, czyli tam, gdzie predykcja Kalmana już stoi; krótkie przerwy zostają jak w tracklab;
- dalsza predykcja prędkością na boisku po CMC (najlepsza hipoteza przy przerwach 25–50 klatek);
- bramka ruchu rozszerzana z czasem od ostatniej aktualizacji, tak żeby o powrocie decydował wygląd.

### Diagnoza: numery koszulek (2026-10-11)

Analiza offline na stałej piątce. Kod bez zmian.
- Tory i surowe odczyty OCR pochodzą z cache obecnego configu: te same `track_id` co w runie `20261010-154834`, sprawdzone.
- Detekcje dopasowane do boxów GT w obrazie: IoU ≥ 0.5, Hungarian per klatka.
- Wyrocznie to zrzuty runu `20261010-154834` z podmienionym numerem, przeliczone `eval-combine`.

**1. Ile kosztują numery.** Wariant pełny, wynik łączny (022 / 028 / 038 / 079 / 086 w nawiasie):

| wariant | pełny HOTA |
|---|---|
| obecny config | 41.0 (49.0 / 37.3 / 56.7 / 19.4 / 39.4) |
| idealne wstrzymanie się: numer toru zostaje, gdy dobry, inaczej brak numeru | 43.6 (58.4 / 39.1 / 56.7 / 22.3 / 39.2) |
| idealne głosowanie z obecnych odczytów: numer z GT, jeśli OCR choć raz go odczytał w torze | 49.5 (66.1 / 52.5 / 58.8 / 27.0 / 44.8) |
| idealny numer toru (większość z GT, także „brak numeru”) | **58.5** (77.0 / 63.3 / 62.7 / 47.3 / 47.0) |
| idealny numer detekcji | 57.6 |
| *bez numerów (dla porównania)* | *59.4* |

- Numery kosztują ok. 17.5 pkt. Lepsze głosowanie z obecnych odczytów może odzyskać do ok. 8.5 pkt. Reszta, ok. 9 pkt, wymaga odczytów, których OCR nie daje.
- „Idealny numer detekcji” wychodzi niżej niż „idealny numer toru”, bo detekcje niedopasowane do GT zostają z numerem toru.

**2. Zawodnicy bez numeru w GT.** 11% detekcji zawodników w GT piątki nie ma numeru: 20% w 022 i 079, 0% w 038. To całe tory (2–4 osoby na sekwencję).
- Metryka porównuje atrybuty jako całość, więc taki zawodnik pasuje tylko do predykcji bez numeru. Tak samo w oryginale.
- Pipeline podaje numer prawie zawsze: bez numeru jest 2–4% predykcji zawodników, w 079 17%. Wystarcza do tego jeden odczyt w torze.

**3. OCR.** Odczyt ma 16% detekcji zawodników z GT, a poprawnych jest 59% odczytów.
Odsetek odczytów rośnie z wysokością boxa: < 80 px: 3%, 80–100 px: 8%, 100–130 px: 12%, > 130 px: 28%. Trafność odczytów nie zależy od wysokości (58–59%).
W 079 OCR czyta tylko 6% detekcji, z czego 26% poprawnie.

**4. Błędy na poziomie toru** (udział detekcji zawodników z GT, tor przypisany osobie większościowej; „odczyty” to mediana liczby odczytów w torze):

| przypadek | udział | odczyty | udział zwycięzcy w głosach |
|---|--:|--:|--:|
| poprawny numer | 48.9% | 83 | 0.76 |
| prawdziwy numer nigdy nie odczytany | 19.8% | 5 | 0.65 |
| ... w tym wygrał fragment prawdziwego numeru (np. „4” zamiast „14”) | 4.1% | 15 | 0.35 |
| prawdziwy odczytany, wygrał inny numer | 8.7% | 21 | 0.44 |
| prawdziwy odczytany, wygrał jego fragment | 6.3% | 61 | 0.52 |
| bez numeru w GT, pipeline podaje numer | 7.4% | 5 | 0.55 |
| brak odczytów w torze | 2.9% | 0 | — |
| bez numeru w GT i w predykcji | 1.8% | 0 | — |

- Błędne numery mają mało dowodów: mediana 5 odczytów wobec 83 przy poprawnych, a zwycięzca ma mniejszy udział w głosach.
  Próg „za mało odczytów → brak numeru” trafiłby w przypadek „bez numeru w GT”. Wyrocznia wstrzymania się pokazuje jednak, że to najwyżej +2.6 pkt.
- Fragmenty numeru (10.4% razem) to typowy błąd odczytu dwucyfrowego numeru jako jednej cyfry. Głosowanie premiujące dwucyfrowy numer zawierający zwycięską cyfrę mogłoby część z tego odzyskać.
- Ok. 8% detekcji należy do torów mieszających dwie osoby (najwięcej w 086: 16%). Głosowanie daje wtedy jeden numer obu.

**Wniosek:** największa dźwignia to sam OCR: 16% odczytów i 59% trafności, a ok. 9 pkt jest poza zasięgiem jakiegokolwiek głosowania.
Tanie poprawki głosowania (fragmenty cyfr, wstrzymanie się przy słabych dowodach, roster jako lista dozwolonych numerów) mają razem zapas do ok. 8.5 pkt.
Uwaga: wszystko to na stałej piątce. Na sekwencjach kontrolnych wariantu pełnego nie ma, bo nie było biegu z OCR.

### Przyspieszenie OCR i wariant pełny na sekwencjach kontrolnych (2026-10-13)

**Przyczyna wolnego OCR.** Nie kształty wejść ani dostrajanie cuDNN. Profil ORT detektora (DBNet, MobileNetV3, 4.7 MB): jedna konwolucja
(depthwise 3×3, `group=96`, `Conv.9`) zajmuje ok. 180 z 210 ms na detekcję; GPU jest wtedy prawie bezczynny (ok. 5%). Rozpoznawanie to ok. 3% czasu.
Powtórzony kształt nie jest szybszy (213 vs 223 ms). Na GPU bez efektu: `cudnn_conv_algo_search` HEURISTIC (211 ms), DEFAULT (ok. 540 ms),
`cudnn_conv_use_max_workspace=1`, strategia i kurczenie areny; `prefer_nhwc` ok. 2.5 s.
**Rozwiązanie:** detektor na CPU (ok. 45 ms na detekcję), opcja `jersey.det_on_cpu` (w kodzie domyślnie `true`, tylko przy providerze CUDA; yaml nietknięty).
Nie próbowano naprawy węzła na GPU (przepisanie grafu, TF32/FP16).

**Odczyty bez zmian.** Harness (kopia logiki OCR) odtworzył odczyty z cache 878/878 (100 klatek 022), CPU = GPU, ta sama pewność.
Pełne 022 przez pipeline (`RERUN=StrongSortModule,TorsoColor,MmocrOnnx`): zrzut bajt w bajt identyczny z `20261010-154834`. OCR: 2.7 → 0.71 s/klatkę (531 s na sekwencję).

**Wariant pełny na 10 sekwencjach kontrolnych** (obecny domyślny config, run `20261013-232134`, własny cache w `<run>/cache` — z OCR):

| sekwencja | pełny | bez nr | pozycja |
|---|--:|--:|--:|
| 023 | 33.9 | 40.9 | 40.8 |
| 031 | 40.0 | 51.0 | 53.9 |
| 034 | 48.3 | 55.0 | 55.5 |
| 041 | 30.6 | 71.4 | 77.1 |
| 044 | 21.5 | 43.8 | 54.3 |
| 048 | 15.0 | 39.3 | 54.4 |
| 050 | 27.1 | 38.1 | 42.3 |
| 056 | 45.3 | 57.7 | 80.8 |
| 082 | 38.1 | 58.7 | 60.2 |
| 092 | 15.4 | 19.4 | 61.9 |
| **łącznie** | **31.5** | **47.2** | **57.7** |

- Bez numerów i sama pozycja zgadzają się z poprzednim biegiem bez OCR (47.2 / 57.7).
- Numery kosztują na kontrolnych ok. 15.7 pkt (piątka: ok. 18.4 przy 41.0 vs 59.4). Najgorzej 041, 044, 048 (mecz 3), gdzie DetA pełnego to 5–10%: prawie żaden zawodnik nie ma poprawnego numeru.
- Od teraz eksperymenty z numerami można puszczać z cache `20261013-232134` na obu zbiorach.
