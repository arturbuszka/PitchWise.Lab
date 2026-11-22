# Geometria kamery nad stołem — dobór parametrów

Dokument wyprowadza, jak z wymiarów stołu i wymagań przetwarzania obrazu (YOLO, śledzenie piłki)
wyznaczyć parametry kamery: obiektyw, wysokość montażu, rozdzielczość, czas naświetlania i liczbę klatek.
Wszystkie długości w mm, czasy w s, prędkości w mm/s, chyba że zaznaczono inaczej.

## 1. Model kamery

Przyjmujemy model kamery otworkowej (*pinhole*) z osią optyczną prostopadłą do boiska, przechodzącą
przez jego środek. Dystorsję obiektywu traktujemy osobno (p. 8) — po jej korekcji model otworkowy obowiązuje.

| Symbol | Znaczenie |
|---|---|
| $f$ | ogniskowa obiektywu [mm] |
| $p$ | rozmiar piksela sensora [mm] |
| $N_x, N_y$ | rozdzielczość sensora w używanym trybie [px] |
| $s_x = N_x p,\ s_y = N_y p$ | wymiary aktywnej części sensora [mm] |
| $h$ | odległość środka rzutowania od płaszczyzny boiska [mm] |
| $f_{px} = f / p$ | ogniskowa w pikselach (wynik kalibracji wewnętrznej, krok 3.5) |

Punkt na boisku w odległości $X$ od osi optycznej rzutuje się na sensor w odległości $x = fX/h$.
Stąd kąty widzenia:

$$\mathrm{HFOV} = 2\arctan\frac{s_x}{2f}, \qquad \mathrm{VFOV} = 2\arctan\frac{s_y}{2f}$$

**Uwaga o trybach pracy:** tryby o niższej rozdzielczości mogą powstawać przez *binning* (łączenie pikseli —
pole widzenia bez zmian, $p$ rośnie) albo przez *crop* (wycięcie — pole widzenia maleje, $p$ bez zmian).
Do wzorów trzeba wziąć $s_x, s_y$ trybu, w którym kamera faktycznie pracuje (karta katalogowa lub kalibracja).

## 2. Wymagane pole widzenia

Kadr musi objąć boisko $L \times W$ z marginesem $m$ z każdej strony (bandy, wejście do bramek, zapas na
niedokładny montaż):

$$W_x = L + 2m, \qquad W_y = W + 2m$$

Kadr kamery ma proporcje $s_x : s_y$. Jeżeli $W_x / W_y > s_x / s_y$, ograniczeniem jest długość stołu
(kadr „wąski”), w przeciwnym razie szerokość. Rozstrzyga to wysokość z p. 3.

## 3. Wysokość montażu

Z podobieństwa trójkątów kadr o szerokości $W_x$ mieści się na sensorze, gdy $f W_x / h \le s_x$, czyli

$$h \ge h_{\min} = \max\left(\frac{f\,W_x}{s_x},\ \frac{f\,W_y}{s_y}\right)$$

Równoważnie, dla zadanej wysokości (np. ograniczonej stelażem) wymagana ogniskowa:

$$f \le \min\left(\frac{h\, s_x}{W_x},\ \frac{h\, s_y}{W_y}\right)$$

W praktyce montuje się kamerę na $h \approx h_{\min}$: wyższy montaż nie poprawia rozdzielczości na boisku
(p. 4), a zwiększa drgania stelaża.

## 4. Rozdzielczość przestrzenna (GSD)

Rozmiar fragmentu boiska odpowiadający jednemu pikselowi (*ground sampling distance*):

$$\mathrm{GSD} = \frac{h\,p}{f} = \frac{h}{f_{px}} \quad [\text{mm/px}]$$

Przy $h = h_{\min}$ (kadr ograniczony szerokością) $\mathrm{GSD} = W_x / N_x$ — **nie zależy od obiektywu
ani wysokości**, tylko od tego, ile pikseli przypada na wymagany kadr.

Detektor nie pracuje na pełnej rozdzielczości: obraz jest skalowany do boku $S$ (np. 640 lub 1280 px,
z zachowaniem proporcji — *letterbox*). Średnica piłki $D$ na wejściu sieci:

$$D_{px} = \frac{D}{\mathrm{GSD}} \cdot \frac{S}{\max(N_x, N_y)}$$

**Kryterium:** detektory jednoetapowe (YOLO) tracą skuteczność dla obiektów poniżej ok. 8–10 px;
przyjmujemy $D_{px} \ge 15$ px z zapasem na rozmycie i zasłonięcia.

## 5. Rozmycie ruchu i czas naświetlania

W czasie naświetlania $t_e$ piłka o prędkości $v$ przesuwa się o

$$b = v\, t_e \quad [\text{mm}], \qquad b_{px} = b / \mathrm{GSD}$$

**Kryterium:** rozmycie nie większe niż połowa średnicy piłki, $b \le D/2$, stąd

$$t_e \le \frac{D}{2\, v_{\max}}$$

Krótki $t_e$ wymaga silnego oświetlenia: ilość światła na piksel jest proporcjonalna do
$t_e / N^2$ ($N$ — liczba przysłony), więc skrócenie $t_e$ dwukrotnie wymaga dwukrotnie jaśniejszego światła.
Źródło światła nie może migać z częstotliwością porównywalną z $1/t_e$ — dlatego LED zasilane prądem stałym,
bez ściemniania PWM.

**Migawka:** przy migawce szczelinowej (*rolling shutter*) wiersze naświetlane są kolejno w czasie odczytu
$t_r$ (rzędu 10–30 ms), więc szybka piłka ulega dodatkowo zniekształceniu o ok. $v\, t_r$. Przy
$v = 10$ m/s to 10–30 cm — dlatego docelowa kamera ma migawkę globalną.

## 6. Liczba klatek na sekundę

Przesunięcie piłki między kolejnymi klatkami:

$$\Delta s = \frac{v}{\mathrm{fps}}$$

Wynikają z tego dwa wymagania:

1. **Śledzenie.** Filtr piłki (`BallFilter`) kojarzy detekcję z przewidywaniem w promieniu bramki
   (*gating*). Promień musi przekraczać błąd przewidywania, który rośnie z $\Delta s$; im mniejsze
   $\Delta s$, tym mniejszy promień i mniej fałszywych skojarzeń. Parametr trafia do `EngineConfig`.
2. **Zdarzenia krótkotrwałe.** Aby piłka była widoczna w co najmniej $k$ klatkach podczas przejścia
   przez strefę o długości $\ell$ (np. przed linią bramki), potrzeba

$$\mathrm{fps} \ge \frac{k\, v_{\max}}{\ell}$$

## 7. Paralaksa

Homografia (krok 3.4) odwzorowuje płaszczyznę boiska $z = 0$. Punkt na wysokości $z$ nad boiskiem,
leżący w odległości $d$ od osi optycznej, rzutuje się tak, jakby leżał w odległości

$$d' = d\,\frac{h}{h - z}, \qquad \text{błąd } \Delta = d' - d = d\,\frac{z}{h - z}$$

**Środek piłki** leży na wysokości promienia $z = r = D/2$. Błąd jest systematyczny i znany, więc koryguje
się go, skalując pozycję z homografii względem punktu pod osią optyczną o czynnik $(h - r)/h$.
Wymaga to znajomości $h$ — trafia do pliku kalibracji.

**Zasłanianie przez figurki.** Figurka o wysokości $z_f$ zasłania z perspektywy kamery pas boiska długości

$$c = z_f\,\frac{d}{h - z_f}$$

po stronie oddalonej od osi. Wyższy montaż (większe $h$) skraca zasłonięcia — to jedyny powód,
by montować kamerę wyżej niż $h_{\min}$ (z węższym obiektywem).

## 8. Dystorsja

Obiektywy szerokokątne wykazują dystorsję beczkową. Model radialny (Brown–Conrady, używany przez OpenCV):

$$x_d = x\,(1 + k_1 \rho^2 + k_2 \rho^4 + k_3 \rho^6) + \text{składniki styczne}, \qquad \rho^2 = x^2 + y^2$$

Współczynniki wyznacza się jednorazowo kalibracją szachownicą (krok 3.5) i usuwa przed homografią.
Korekcja prostuje geometrię, ale nie odzyskuje rozdzielczości: na brzegach kadru szerokokątnego GSD jest
gorsze niż w środku. Dlatego unikamy obiektywów o HFOV powyżej ok. 90°.

## 9. Głębia ostrości

Dla odległości $u \gg f$ głębia ostrości wynosi w przybliżeniu

$$\mathrm{DOF} \approx \frac{2 N c\, u^2}{f^2}$$

gdzie $c$ — dopuszczalny krążek rozmycia (przyjmujemy $c = 2p$). Interesujący zakres głębokości to
ok. 0–110 mm nad boiskiem (piłka, figurki), więc warunek $\mathrm{DOF} \gg 110$ mm jest w praktyce zawsze
spełniony (p. 10).

## 10. Przykład: stół docelowy z kamerą AR0234

Dane: boisko $1200 \times 700$, margines $m = 50$ → $W_x = 1300$, $W_y = 800$.
Sensor AR0234: $1920 \times 1200$ px, $p = 3{,}0$ µm → $s_x = 5{,}76$, $s_y = 3{,}60$ mm. Obiektyw $f = 6$ mm.
Piłka $D = 35$ mm. Założona maks. prędkość piłki $v_{\max} = 10$ m/s $= 10^4$ mm/s.

| Wielkość | Obliczenie | Wynik |
|---|---|---|
| HFOV, VFOV | $2\arctan(5{,}76/12)$, $2\arctan(3{,}6/12)$ | 51,3°, 33,4° |
| $h_{\min}$ | $\max(6 \cdot 1300/5{,}76,\ 6 \cdot 800/3{,}6)$ | $\max(1354, 1333) = 1354$ mm |
| GSD (pełna rozdzielczość) | $1300/1920$ | 0,68 mm/px |
| Piłka na wejściu YOLO, $S = 1280$ | $35/0{,}68 \cdot 1280/1920$ | 34 px |
| Piłka na wejściu YOLO, $S = 640$ | jw. $\cdot\, 640/1920$ | 17 px |
| Maks. czas naświetlania | $35 / (2 \cdot 10^4)$ | 1,75 ms |
| Rozmycie przy $t_e = 1$ ms | $10^4 \cdot 10^{-3}$ | 10 mm = 15 px |
| $\Delta s$ przy 120 fps | $10^4 / 120$ | 83 mm = 2,4 $D$ |
| Błąd paralaksy środka piłki w narożniku ($d = \sqrt{600^2 + 350^2} = 695$) | $695 \cdot 17{,}5 / (1354 - 17{,}5)$ | 9,1 mm |
| Zasłonięcie przez figurkę ($z_f \approx 105$) przy bramce ($d = 600$) | $105 \cdot 600 / (1354 - 105)$ | 50 mm |
| DOF ($N = 2$, $c = 6$ µm, $u = 1354$) | $2 \cdot 2 \cdot 0{,}006 \cdot 1354^2 / 36$ | ok. 1,2 m |

Wnioski: $S = 640$ spełnia kryterium $D_{px} \ge 15$ z niewielkim zapasem, $S = 1280$ z dużym.
Bez korekcji paralaksy pozycja piłki przy narożnikach jest obarczona błędem ok. 1 cm.

## 11. Procedura doboru

1. Wyznacz kadr $W_x, W_y$ (p. 2).
2. Z ograniczeń mechanicznych (sztywność stelaża, sufit, światło) wybierz zakres wysokości $h$.
3. Dobierz ogniskową $f$ (p. 3) — dla obiektywów o zmiennej ogniskowej ustaw ją na miejscu.
4. Sprawdź rozdzielczość: $D_{px} \ge 15$ przy rozmiarze wejścia detektora (p. 4).
5. Wyznacz $t_e \le D / 2v_{\max}$ i dobierz oświetlenie (p. 5).
6. Sprawdź fps pod kątem śledzenia i zdarzeń (p. 6).
7. Oszacuj błąd paralaksy i zasłonięcia (p. 7); jeśli zasłonięcia są zbyt duże — podnieś kamerę i zwęź obiektyw.
8. Po montażu: kalibracja wewnętrzna (szachownica, $f_{px}$, dystorsja) i zewnętrzna (znaczniki ArUco, homografia, $h$).

## Parametry trafiające do konfiguracji

| Parametr | Źródło | Użycie |
|---|---|---|
| $f_{px}$, $k_1..k_3$, składniki styczne | kalibracja szachownicą (3.5) | korekcja dystorsji |
| homografia boisko ↔ obraz | znaczniki ArUco (3.4) | piksele → mm |
| $h$ | kalibracja zewnętrzna lub pomiar | korekcja paralaksy |
| $D$ (średnica piłki) | `table.json` | korekcja paralaksy, kryteria detekcji |
| $t_e$, fps | ustawienia kamery (6.1) | — |
| promień bramki kojarzenia | z $\Delta s$ | `EngineConfig` (3.6) |
