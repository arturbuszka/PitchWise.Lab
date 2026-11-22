# Ocena detektora piłki (krok 2.3)

Model `foosball-yolo11n-v4`, test: 389 klatek z filmu 5, którego nie ma w treningu. Mały stół testowy,
iPhone 13 mini z góry, 1920×1080. Odtworzenie:

```bash
~/.venvs/pitchwise-export/bin/python tools/training/ball_eval.py --model best-v4.pt \
    --images <v4>/test/images --glob 'training_iphone13mini_5_*.jpg' --out /tmp/ball_eval
```

## Recall piłki według przypadku

Próg pewności 0.25, trafienie przy IoU ≥ 0.5. Etykiety nie mają tagów, więc przypadek wynika z kształtu ramki:
szybki strzał to ramka wydłużona przez rozmazanie (dłuższy bok / krótszy ≥ 1.3), częściowe zasłonięcie to ramka
mniejsza niż 0.6 mediany.

| Przypadek | Piłki | Wykryte | Recall |
|---|---|---|---|
| swobodna | 331 | 331 | 1.000 |
| szybki strzał | 22 | 22 | 1.000 |
| częściowo zasłonięta | 6 | 6 | 1.000 |
| razem | 359 | 359 | 1.000 |

Dla porównania `yolo val` (próg 0.001, krzywa PR): piłka P 0.981, R 0.986, mAP50 0.994, mAP50-95 0.906.

## Fałszywe piłki: 15, z czego żadna nie jest pomyłką z figurką

Przejrzane na oko, wszystkie leżą na prawdziwej piłce:

- 9: druga ramka na tej samej piłce, częściowo zasłoniętej dłonią lub figurką. Obie ramki pokrywają się za słabo, żeby
  NMS usunął jedną z nich.
- 6: piłka pod figurką albo w dłoni, której nie oznaczono. Etykieta jest niespójna z regułą „widać co najmniej ~1/3”.

## Decyzja

Model wystarcza do kroku 3. Nie dooznaczamy danych teraz.

- Detektor (3.2) zwraca z klatki najwyżej jedną piłkę, tę o najwyższej pewności, bo na stole jest jedna. To usuwa
  podwójne ramki.
- Przy następnej wersji danych poprawić etykiety piłki pod figurką.
- Ograniczenia oceny: tylko 22 szybkie strzały i 6 zasłonięć, jeden stół, jedno światło, jedna kamera. Na dużym stole
  z docelową kamerą ocenę trzeba powtórzyć na nowym nagraniu.
