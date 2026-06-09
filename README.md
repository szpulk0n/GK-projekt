
# GK-projekt — wizualizacja 3D (OpenGL, Python)

## Krótkie wprowadzenie

Projekt to aplikacja demonstracyjna napisana w Pythonie, wykorzystująca OpenGL (PyOpenGL) i GLFW do renderowania sceny 3D zawierającej Ziemię, kometę, system cząsteczek oraz skybox. Kod jest rozbity na moduły odpowiadające za okno, kamerę, fizykę i shadery.

## Wymagania

- System: macOS / Linux / Windows (zalecane macOS lub Linux dla prostszej konfiguracji)
- Python 3.10+ (użyj wirtualnego środowiska)
- Zainstalowane zależności z `requirements.txt`:

	- PyOpenGL>=3.1.7
	- glfw>=2.7.0
	- numpy>=1.26.0

Plik z wymaganiami znajduje się w repozytorium: [requirements.txt](requirements.txt).

## Instalacja (krok po kroku)

1. Sklonuj repozytorium (jeśli jeszcze tego nie zrobiłeś):

```bash
git clone <repo-url>
cd GK-projekt
```

2. Utwórz i aktywuj wirtualne środowisko (przykład dla macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Na Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

4. (Opcjonalnie) Jeśli masz problemy z instalacją `glfw` lub PyOpenGL, upewnij się, że masz zainstalowane narzędzia do budowy i odpowiednie nagłówki systemowe. Na macOS można użyć Homebrew do instalacji niezbędnych narzędzi.

## Uruchomienie

Główny punkt wejścia to `main.py`. Po aktywowaniu środowiska uruchom:

```bash
python main.py
```

W razie potrzeby możesz uruchomić z poziomu IDE, upewniając się, że interpreter używa wirtualnego środowiska `.venv`.

## Sterowanie i opcje

Poniżej znajduje się szczegółowe, domyślne mapowanie klawiszy i działań wejściowych. Konkretne mapowania mogą się różnić w zależności od implementacji w `camera.py`, `window.py` lub `main.py` — w razie wątpliwości sprawdź te pliki.

### Myszka


### Ruch kamery (klawiatura)


### Przyciski funkcyjne i skróty


### Uwagi


Jeśli chcesz poznać dokładne sterowanie, otwórz [src/main.py](src/main.py) i [src/window.py](src/window.py).
## Sterowanie i opcje (rzeczywiste mapowanie z kodu)

Poniżej znajduje się dokładne mapowanie klawiszy i wejścia odczytane bezpośrednio z `src/main.py`, `src/window.py` i `src/camera.py`.

### Myszka

- Ruch myszy (kursor wyłączony): obrót kamery (yaw/pitch) — obsługiwane przez `camera.process_mouse_movement` (kursor jest ustawiony w trybie `glfw.CURSOR_DISABLED`).

### Klawiatura — ruch kamery i globalne akcje

- `W`: ruch do przodu (`FORWARD`)
- `S`: ruch do tyłu (`BACKWARD`)
- `A`: ruch w lewo (`LEFT`)
- `D`: ruch w prawo (`RIGHT`)
- `Esc`: zamknięcie aplikacji
- Strzałka w górę (`UP`): zwiększanie skali czasu/szybkości symulacji (sterowane w `process_input`)
- Strzałka w dół (`DOWN`): zmniejszanie skali czasu/szybkości symulacji

### Przełączanie scenariuszy i kontrolki scenariusza 4

- `1`, `2`, `3`, `4`: przełączenie aktywnego scenariusza

W scenariuszu 4 (kiedy jest aktywny):

- `Left Shift` / `Right Shift`: przełączenie aktywnego suwaka (cyklicznie)
- `Left Arrow` / `Right Arrow`: zmniejsz/zmień wartość aktywnego suwaka (z krótkim cooldownem)
- `Enter`: uruchom (wystrzel) kometę z aktualnymi wartościami suwaków (size, speed, mass)
- `R`: zresetuj kometę i suwak do wartości domyślnych

## Struktura projektu

- `src/` — kod źródłowy aplikacji
	- `main.py` — punkt wejścia aplikacji
	- `window.py` — obsługa okna i pętli renderującej
	- `camera.py` — logika kamery i widoku
	- `physics.py` — prosta symulacja ruchu i sił
	- `earth.py`, `comet.py` — moduły odpowiadające za obiekty sceny
	- `shader.py` — pomocnicze funkcje ładowania i kompilowania shaderów
	- `skybox.py` — obsługa skyboxa
	- `shaders/` — pliki shaderów GLSL (`*.vert`, `*.frag`)

## Co robi każdy moduł (szybki przegląd)

- `window.py`: inicjalizuje GLFW, tworzy kontekst OpenGL i pętlę renderującą.
- `camera.py`: oblicza macierze widoku i projekcji; obsługuje wejście użytkownika.
- `shader.py`: ładuje źródła shaderów, kompiluje i obsługuje uniformy.
- `physics.py`: aktualizuje pozycję i prędkość obiektów (np. komety, cząstek).
- `earth.py` / `comet.py`: definiują siatki, tekstury i logikę renderowania obiektów.

## Shadery

Shadery znajdują się w `src/shaders/`. Jeśli zmodyfikujesz pliki `.vert` lub `.frag`, zrestartuj aplikację, aby zobaczyć zmiany.

## Debugowanie i najczęstsze problemy

- `ModuleNotFoundError` po uruchomieniu: upewnij się, że aktywowałeś wirtualne środowisko i zainstalowałeś zależności (`pip install -r requirements.txt`).
- Problemy z inicjalizacją GLFW: sprawdź, czy system ma dostęp do kontekstu OpenGL (na serwerach bez GPU może nie działać).
- Błędy shaderów: komunikaty kompilatora shaderów wypisywane są zwykle w konsoli — otwórz pliki `.vert`/`.frag`, sprawdź składnię GLSL.

## Rozszerzanie projektu

- Dodawanie nowych obiektów: utwórz nowy moduł wzorowany na `comet.py` lub `earth.py`, dodaj inicjalizację w `main.py`.
- Eksperymenty z shaderami: edytuj pliki w `src/shaders/` i dostosuj `shader.py` do przekazywania nowych uniformów.

## Testy i walidacja

Projekt nie zawiera zautomatyzowanych testów. Ręczne sprawdzenie polega na uruchomieniu `python main.py` i obserwacji czy scena renderuje się poprawnie.


