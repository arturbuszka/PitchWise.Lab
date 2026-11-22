# Dane do treningu YOLO (piłkarzyki)

Zbiór żyje w Roboflow (projekt `smartfoosball`), eksport poza repo:
`~/Studia/PROJEKTP/data_set_training/versions/<wersja>/` w formacie YOLOv11. Opis nagrań: `docs/foosball/recordings.md`.

## Klasy

| id | Klasa | Ramka |
|---|---|---|
| 0 | `ball` | Wokół całej piłki, także częściowo zasłoniętej, jeśli widać co najmniej ~1/3. Mniej — nie oznaczać. |
| 1 | `figure` | Wokół całej widocznej figurki z góry (jak robi auto-label): obrócona figurka ma większą ramkę. Kod bierze z niej tylko pozycję wzdłuż drążka. |

Boiska, band ani drążków nie oznaczamy.

## Jak powstaje wersja

1. Nagranie (iPhone, slow-mo) → kopia H.264: `tools/training/convert_to_mp4.sh <pliki .MOV>` (wynik w `mp4/` obok).
   Na Linuksie pliki z iPhone'a najpewniej przez `ifuse` (`~/Projects/scripts/iphone.sh mount`), nie przez menedżer plików.
2. Upload do Roboflow, próbkowanie ~3 klatki/s (przy 240 fps sąsiednie klatki są prawie identyczne).
3. Auto-label, potem ręczny przegląd: żółte figurki brane za piłkę, pominięte rozmazane piłki.
4. Wersja: podział 70/20/10, preprocessing tylko Auto-Orient (bez Resize: rozciąga 16:9 i zmniejsza piłkę),
   bez augmentacji (robi je Ultralytics).
5. Eksport YOLOv11, zip do katalogu wersji.

Test to jeden cały film, którego nie ma w train ani w valid (od v4: film 5). Losowy podział klatek z tego
samego filmu zawyża wynik, bo sąsiednie klatki są prawie identyczne.

## Wersje

| Wersja | Nagrania | Obrazy (train/val/test) | Piłki / figurki (train) |
|---|---|---|---|
| v1 | `IMG_0426` | 178 / 51 / 25 | 177 / 3044 |
| v4 | filmy 1–5 (film 5 = test) | 1085 / 276 / 400 | 919 / 19175 |

## Modele

| Model | Dane | Trening | Test: mAP50 / mAP50-95 (ball, figure) | Inferencja |
|---|---|---|---|---|
| v1-yolo11n | v1 | Colab T4, imgsz 1280, 100 epok, batch 16 | 0.995 / 0.907 (0.907, 0.908), P 0.9997, R 0.9995 | 12 ms/obraz na T4 przy 1280 |
| v4-yolo11n | v4 | jak v1 | 0.994 / 0.901 (0.906, 0.897), piłka P 0.981 R 0.986 | 256 ms/obraz na CPU laptopa |

v1: wynik zawyżony — test to klatki z tego samego filmu co trening. Prawdziwa ocena wymaga osobnego nagrania (krok 2.3).
v4: test tylko na 389 klatkach filmu 5 (11 klatek z filmów 1–2, które wpadły do test, pominięte).
Rozbicie na trudne przypadki: `docs/foosball/eval.md`, skrypt `tools/training/ball_eval.py`.
Wagi lokalnie: `~/Studia/PROJEKTP/models/best-v4.pt`; na Google Drive w `MyDrive/foosball/runs/`.
