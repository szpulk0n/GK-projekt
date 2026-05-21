# Protoplanetary Disk Simulation
## Grafika Komputerowa - Projekt Zaliczeniowy

Projekt symuluje powstawanie układu słonecznego (tzw. dysk protoplanetarny). Wykorzystuje czysty, nowoczesny profil PyOpenGL (Core 3.3+), bibliotekę Numpy do optymalizacji wektorowych obliczeń fizycznych, Instanced Rendering, własne shadery (GLSL) oraz proceduralnie generowany Skybox.

### Jak uruchomić

1. Aktywuj wirtualne środowisko (jeśli używasz):
   ```bash
   source venv/bin/activate
   ```
2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
3. Uruchom program:
   ```bash
   python src/main.py
   ```

### Sterowanie i Interakcja

**Kamera:**
- Myszka (Free-look) - rozglądanie się
- `W`, `A`, `S`, `D` - swobodne latanie w przestrzeni 3D

**Kontrola czasu (Czasoprzestrzeń):**
- `Strzałka w górę` - płynne przyspieszanie upływu czasu
- `Strzałka w dół` - płynne zwalnianie / zatrzymywanie czasu

**Przełączanie Scenariuszy:**
- Klawisz `1`: **Pojedyncza Gwiazda** 
  * Klasyczny dysk protoplanetarny. 10 000 cząsteczek orbituje wokół potężnego, centralnego źródła grawitacji.
- Klawisz `2`: **Układ Podwójny (Binary Star System)**
  * Dwie supermasywne gwiazdy tańczą wokół siebie na środku mapy. Cząsteczki orbitują w niezwykle skomplikowanym polu grawitacyjnym, tworząc niesamowite wzory i "rozedrgane" trajektorie lotu (Szybki algorytm $O(N)$).
- Klawisz `3`: **Supernowa (Wybuch)**
  * Cząsteczki startują z samego centrum i gwałtownie eksplodują we wszystkich kierunkach z dużą prędkością. Są powstrzymywane powoli tylko przez delikatną grawitację pozostałego rdzenia oraz "opór gazu kosmicznego" (Szybki algorytm $O(N)$).

---

### Detale techniczne
- Cząstki bliżej gwiazdy świecą na żółto, średnie na pomarańczowo, odległe na niebiesko (obliczane bezpośrednio w Vertex Shaderze).
- Zamiast pobierać gigabajty tekstur z internetu, symulator generuje proceduralny sześcian (Cubemap) wypełniony różnokolorowymi gwiazdami za pomocą tablic Numpy i ładuje go do pamięci GPU w ułamku sekundy.
