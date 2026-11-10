# Praca nad PitchWise.Foosball

Budujemy system statystyk dla piłkarzyków 1v1 w barze. Docelowa architektura i plan budowy są w
[`docs/foosball/ARCHITECTURE.md`](../../docs/foosball/ARCHITECTURE.md). Przeczytaj go na początku każdej sesji. To jedyne źródło prawdy o tym,
co i w jakiej kolejności robimy.

## Najważniejsza zasada: małe kroki

Pracujemy **wyłącznie po jednym kroku z planu** (np. „3.2 Detektor YOLO”). Nigdy nie wychodzisz poza
zakres bieżącego kroku, nawet jeśli następny wydaje się oczywisty.

Każdy krok przebiega tak:

1. **Plan, zanim napiszesz kod.** Napisz krótko:
   - cel kroku (z planu),
   - jakie pliki utworzysz lub zmienisz (pełna lista),
   - jakie typy i metody publiczne powstaną (nazwy + jedno zdanie, co robią),
   - jak sprawdzimy, że działa (test, komenda).
   Potem **czekaj na moją akceptację**.
2. **Implementacja.** Tylko to, co było w zaakceptowanym planie.
   - Jeśli zmiana przekracza ~150 linii, podziel krok na mniejsze podkroki (3.2a, 3.2b...) i zaproponuj to w planie.
   - Jeśli w trakcie okaże się, że plan był zły, zatrzymaj się i powiedz, zamiast improwizować.
3. **Weryfikacja.** Uruchom `dotnet build PitchWise.slnx` i `dotnet test PitchWise.slnx`. Pokaż wynik.
4. **Podsumowanie dla mnie** (krótko):
   - co powstało i gdzie,
   - jak to działa w 3–5 zdaniach (przepływ danych, nie opis każdej linii),
   - na co zwrócić uwagę przy czytaniu kodu,
   - czy zostało coś niedokończonego.
5. **Commit.** Po mojej akceptacji jeden commit na krok, wiadomość w formacie `foosball: 3.2 detektor YOLO`.
   Odhacz krok w [`docs/foosball/ARCHITECTURE.md`](../../docs/foosball/ARCHITECTURE.md) w tym samym commicie.
6. **Stop.** Nie zaczynaj następnego kroku, dopóki o to nie poproszę.

## Granice architektury

- Nie modyfikuj `PitchWise.Engine` ani `PitchWise.GameState`. Jeśli coś w nich przeszkadza, opisz problem i zaproponuj rozwiązanie, ale nie zmieniaj ich sam.
- Biblioteki Foosball nie zależą od `PitchWise.GameState` ani `PitchWise.App.Core`.
- Kontraktem między wizją a logiką gry jest `FrameObservation`. Piksele nie wychodzą poza `PitchWise.Foosball.Vision`.
- Wyjście offline ma ten sam format co komenda `analyze` (`result.json`, `world_state.jsonl`, wideo z nakładką).
- Wszystkie progi i parametry trafiają do konfiguracji, nie są wpisane na sztywno w kodzie.

## Styl kodu

- Trzymaj się konwencji istniejącego repo (.NET 10, nullable, rekordy, komentarze `///` wyjaśniające „dlaczego”, a nie „co”).
- Wolę prosty, czytelny kod od sprytnego. Bez abstrakcji „na zapas”: interfejs tylko tam, gdzie plan przewiduje drugą implementację (źródło klatek, detektor).
- Każdy krok kończy się testem. Testy wymagające modeli lub nagrań mają być pomijane, gdy plików brak (jak w reszcie repo).
- Nie dodawaj nowych pakietów NuGet bez zapytania.

## Komunikacja

- Odpowiadaj po polsku, krótko.
- Gdy czegoś nie wiesz (np. wymiarów stołu, formatu danych z CVAT), zapytaj zamiast zgadywać.
- Gdy proszę o coś, co wychodzi poza bieżący krok, przypomnij, na jakim kroku jesteśmy, i zapytaj, czy dopisać to do planu.
