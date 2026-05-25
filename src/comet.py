"""
Modul komety/asteroidy dla scenariusza 4.
Kometa ma ksztalt elipsoidy, ogon pylu, i leci w kierunku Ziemi.
Po uderzeniu generuje animacje zderzenia.
"""

import OpenGL.GL as gl
import numpy as np
import math
import os
from shader import Shader
from earth import Earth


# ─────────────────────────────────────────────
#  Pomocnicze: siatka elipsoidy (trojkaty)
# ─────────────────────────────────────────────

def _make_ellipsoid_mesh(rx: float = 1.0, ry: float = 0.65, rz: float = 0.8,
                          stacks: int = 32, slices: int = 32):
    """Elipsoida jako siatka trojkatow (analogicznie do sfery w earth.py)."""
    vertices = []
    indices  = []

    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            x = rx * sin_phi * cos_theta
            y = ry * cos_phi
            z = rz * sin_phi * sin_theta

            # Normalna elipsoidy
            nx = sin_phi * cos_theta / rx
            ny = cos_phi             / ry
            nz = sin_phi * sin_theta / rz
            length = math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-9
            nx /= length; ny /= length; nz /= length

            vertices.extend([x, y, z, nx, ny, nz])

    for i in range(stacks):
        for j in range(slices):
            tl = i * (slices + 1) + j
            tr = tl + 1
            bl = (i + 1) * (slices + 1) + j
            br = bl + 1
            indices.extend([tl, bl, tr])
            indices.extend([tr, bl, br])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


# ─────────────────────────────────────────────
#  Czasteczki ogona / gruzu
# ─────────────────────────────────────────────

class ParticleSystem:
    """GPU-side system czasteczek (pozycja, kolor, rozmiar)."""

    MAX = 8000

    def __init__(self, shader: Shader):
        self.shader = shader
        self.count  = 0

        # CPU bufory
        self.positions = np.zeros((self.MAX, 3), dtype=np.float32)
        self.colors    = np.zeros((self.MAX, 3), dtype=np.float32)
        self.sizes     = np.zeros(self.MAX,       dtype=np.float32)
        self.velocities= np.zeros((self.MAX, 3), dtype=np.float32)
        self.lifetimes = np.zeros(self.MAX,       dtype=np.float32)
        self.max_life  = np.ones(self.MAX,        dtype=np.float32)

        # VAO / VBO
        self.vao = gl.glGenVertexArrays(1)
        self.pos_vbo  = gl.glGenBuffers(1)
        self.col_vbo  = gl.glGenBuffers(1)
        self.size_vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)

        # aPos (location=0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pos_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # aColor (location=1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.col_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        # aSize (location=2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.size_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.sizes.nbytes, self.sizes, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

    def emit(self, pos: np.ndarray, vel: np.ndarray, color: np.ndarray,
             size: float, lifetime: float, count: int = 1):
        for _ in range(count):
            if self.count >= self.MAX:
                break
            i = self.count
            jitter = np.random.randn(3).astype(np.float32) * 0.5
            self.positions[i] = pos + jitter
            jitter_v = np.random.randn(3).astype(np.float32)
            self.velocities[i] = vel + jitter_v
            self.colors[i]    = np.clip(color + np.random.randn(3)*0.05, 0, 1)
            self.sizes[i]     = size * (0.7 + np.random.rand() * 0.6)
            self.lifetimes[i] = lifetime * (0.8 + np.random.rand() * 0.4)
            self.max_life[i]  = self.lifetimes[i]
            self.count += 1

    def update(self, dt: float):
        if self.count == 0:
            return
        n = self.count
        self.lifetimes[:n] -= dt
        alive = self.lifetimes[:n] > 0
        # Ruch i grawitacja sloneczna (prosta)
        self.positions[:n] += self.velocities[:n] * dt
        self.velocities[:n, 1] -= 0.5 * dt  # slaba grawitacja w dol

        # Fade koloru
        t = self.lifetimes[:n] / (self.max_life[:n] + 1e-9)
        # Sciemnij ku czerwieni
        self.colors[:n, 0] = np.clip(t * 1.2, 0, 1)
        self.colors[:n, 1] = np.clip(t * 0.4, 0, 1)
        self.colors[:n, 2] = np.clip(t * 0.1, 0, 1)
        self.sizes[:n] *= 0.998

        # Usun martwe
        mask = np.where(alive)[0]
        dead_mask = np.where(~alive)[0]
        if len(dead_mask) > 0:
            keep = len(mask)
            self.positions[:keep]  = self.positions[mask]
            self.velocities[:keep] = self.velocities[mask]
            self.colors[:keep]     = self.colors[mask]
            self.sizes[:keep]      = self.sizes[mask]
            self.lifetimes[:keep]  = self.lifetimes[mask]
            self.max_life[:keep]   = self.max_life[mask]
            self.count = keep

    def upload(self):
        if self.count == 0:
            return
        n = self.count
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pos_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.positions[:n].nbytes, self.positions[:n])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.col_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.colors[:n].nbytes, self.colors[:n])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.size_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.sizes[:n].nbytes, self.sizes[:n])

    def draw(self, view: np.ndarray, projection: np.ndarray):
        if self.count == 0:
            return
        self.upload()
        self.shader.use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader.program_id, "view"),
                              1, gl.GL_FALSE, view.T.copy())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader.program_id, "projection"),
                              1, gl.GL_FALSE, projection.T.copy())
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)


# ─────────────────────────────────────────────
#  Fragmenty orbitalne po zderzeniu
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  Fragmenty orbitalne po zderzeniu
#  (styl identyczny z Scenariuszem 1 – orbity
#   kolowe wokol Ziemi, grawitacja centralna)
# ─────────────────────────────────────────────

class OrbitalDebris:
    """
    Fragmenty komety krazace wokol Ziemi po uderzeniu, renderowane jako instancjonowane sześciany!
    Fizyka odskoku uwzględniająca kierunek wejścia, działająca pod wpływem
    grawitacji kuli ziemskiej z zachowaniem kolizji z planetą (sprężyście).
    """

    G  = 0.5
    GM = 0.5 * 10000.0 * 0.01

    MAX = 1000

    def __init__(self, shader: Shader):
        self.shader = shader
        self.n = 0

        self.pos = np.zeros((self.MAX, 3), dtype=np.float32)
        self.vel = np.zeros((self.MAX, 3), dtype=np.float32)

        # ── Cube vertices (identycznie jak w scenariuszu 1, 2, 3) ──
        vertices = np.array([
            # Back face
            -0.5,-0.5,-0.5, 1.0,0.0,0.0,  0.5,-0.5,-0.5, 0.0,1.0,0.0,  0.5,0.5,-0.5, 0.0,0.0,1.0,
             0.5, 0.5,-0.5, 0.0,0.0,1.0, -0.5,0.5,-0.5, 1.0,1.0,0.0, -0.5,-0.5,-0.5, 1.0,0.0,0.0,
            # Front face
            -0.5,-0.5,0.5, 1.0,0.0,0.0,  0.5,-0.5,0.5, 0.0,1.0,0.0,  0.5,0.5,0.5, 0.0,0.0,1.0,
             0.5, 0.5,0.5, 0.0,0.0,1.0, -0.5,0.5,0.5, 1.0,1.0,0.0, -0.5,-0.5,0.5, 1.0,0.0,0.0,
            # Left face
            -0.5,0.5,0.5, 1.0,0.0,1.0, -0.5,0.5,-0.5, 0.0,1.0,1.0, -0.5,-0.5,-0.5, 1.0,1.0,1.0,
            -0.5,-0.5,-0.5,1.0,1.0,1.0,-0.5,-0.5,0.5, 0.0,0.0,0.0, -0.5,0.5,0.5, 1.0,0.0,1.0,
            # Right face
            0.5,0.5,0.5, 1.0,0.0,1.0,  0.5,0.5,-0.5, 0.0,1.0,1.0,  0.5,-0.5,-0.5, 1.0,1.0,1.0,
            0.5,-0.5,-0.5,1.0,1.0,1.0, 0.5,-0.5,0.5, 0.0,0.0,0.0,  0.5,0.5,0.5, 1.0,0.0,1.0,
            # Bottom face
            -0.5,-0.5,-0.5,0.5,0.5,0.5, 0.5,-0.5,-0.5,0.5,0.0,0.0, 0.5,-0.5,0.5,0.0,0.5,0.0,
             0.5,-0.5, 0.5,0.0,0.5,0.0,-0.5,-0.5,0.5,0.0,0.0,0.5, -0.5,-0.5,-0.5,0.5,0.5,0.5,
            # Top face
            -0.5,0.5,-0.5,0.5,0.5,0.5, 0.5,0.5,-0.5,0.5,0.0,0.0,  0.5,0.5,0.5,0.0,0.5,0.0,
             0.5,0.5, 0.5,0.0,0.5,0.0,-0.5,0.5,0.5,0.0,0.0,0.5, -0.5,0.5,-0.5,0.5,0.5,0.5,
        ], dtype=np.float32)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # attr 0: pozycje siatki sześcianu
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # attr 1: kolory siatki (chociaż basic.vert korzysta potem z odległości, trzeba to przepuścić)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, gl.ctypes.c_void_p(3*vertices.itemsize))
        gl.glEnableVertexAttribArray(1)

        self.instance_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, gl.GL_DYNAMIC_DRAW)

        # attr 2: instanced pozycji (aOffset)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 3*self.pos.itemsize, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribDivisor(2, 1)

        gl.glBindVertexArray(0)

    def reset(self):
        self.n = 0

    def spawn(self, impact_point: np.ndarray, earth_radius: float,
              comet_size: float, comet_mass: float, comet_speed: float, comet_direction: np.ndarray):
        """Wyrzut kostek po zderzeniu, przeciwnie do lotu komety i jako ejecta krateru."""
        n = min(int(100 + comet_mass * 15 + comet_size * 5), self.MAX)

        up = impact_point
        # Wektor odskoku to mix wektora zderzenia ("w górę" krateru) z odwróconym ruchem komety
        rebound_base = np.copy(up) * 0.5 - comet_direction * 0.5
        v_norm = np.linalg.norm(rebound_base)
        if v_norm > 0: rebound_base /= v_norm

        arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tangent1 = np.cross(up, arbitrary)
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(up, tangent1)

        v_escape = np.sqrt(2 * self.GM / earth_radius)
        base_pos = up * (earth_radius)

        for i in range(n):
            u = np.random.uniform(-1, 1)
            v = np.random.uniform(-1, 1)
            offset = (tangent1 * u + tangent2 * v) * (comet_size * 0.4)
            self.pos[i] = base_pos + offset

            # Przewaga wyrzutu odskakującego wraz z domieszką tangensową, żeby orbitowały
            kick_out = rebound_base * np.random.uniform(0.5, 1.3)
            kick_tangent = (tangent1 * np.random.uniform(-1, 1) + tangent2 * np.random.uniform(-1, 1)) * 1.5
            
            dir_vec = kick_out + kick_tangent
            
            # Ich prędkość jest bazowana na sile zderzenia (prędkości ucieczki * mnożnik)
            spd = v_escape * np.random.uniform(0.3, 1.1)
            
            self.vel[i] = (dir_vec / np.linalg.norm(dir_vec)) * spd

        self.n = n

    def update(self, dt: float):
        if self.n == 0:
            return
        n = self.n
        pos = self.pos[:n]
        r2  = np.sum(pos ** 2, axis=1, keepdims=True) + 1e-9
        r1  = np.sqrt(r2)
        acc = -self.GM * pos / (r1 * r2)

        self.vel[:n] += acc * dt
        self.pos[:n] += self.vel[:n] * dt

        # Kolizja z Ziemią
        hits = (r1.flatten() < 10.0)
        if np.any(hits):
            normal = pos[hits] / r1[hits]
            dot = np.sum(self.vel[:n][hits] * normal, axis=1, keepdims=True)
            
            moving_in = (dot.flatten() < 0)
            hits_in = hits.copy()
            hits_in[hits] = moving_in
            
            if np.any(hits_in):
                normal_in = pos[hits_in] / r1[hits_in]
                dot_in = np.sum(self.vel[:n][hits_in] * normal_in, axis=1, keepdims=True)
                
                # Sprężysty odskok ze stratą (0.8)
                self.vel[:n][hits_in] -= 1.8 * dot_in * normal_in
                # Korekcja pozycji
                self.pos[:n][hits_in] += normal_in * (10.01 - r1[hits_in])

    def _upload(self):
        n = self.n
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.pos[:n].nbytes, self.pos[:n])

    def draw(self, view: np.ndarray, projection: np.ndarray):
        if self.n == 0:
            return
        self._upload()
        self.shader.use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader.program_id, "view"),
                              1, gl.GL_FALSE, view.T.copy())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader.program_id, "projection"),
                              1, gl.GL_FALSE, projection.T.copy())
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, self.n)
        gl.glBindVertexArray(0)


# ─────────────────────────────────────────────
#  Stan komety
# ─────────────────────────────────────────────

class CometState:
    IDLE    = "idle"       # Czeka na uruchomienie
    FLYING  = "flying"     # Leci w kierunku Ziemi
    IMPACT  = "impact"     # Animacja zderzenia
    DONE    = "done"       # Koniec animacji


class Comet:
    """
    Kometa z interaktywna konfiguracją (rozmiar, predkosc, masa).
    Parametry:
        size   : promien bazowy [1-10]
        speed  : predkosc [km/s scal] [5-80]
        mass   : masa [1e9 - 1e15 kg scal]
    """

    EARTH_RADIUS = 10.0    # promien Ziemi w jednostkach sceny
    START_DIST   = 120.0   # odleglosc startowa komety

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.comet_shader = Shader(
            os.path.join(base_dir, "shaders", "comet.vert"),
            os.path.join(base_dir, "shaders", "comet.frag"),
        )
        self.particle_shader = Shader(
            os.path.join(base_dir, "shaders", "particle.vert"),
            os.path.join(base_dir, "shaders", "particle.frag"),
        )
        self.basic_shader = Shader(
            os.path.join(base_dir, "shaders", "basic.vert"),
            os.path.join(base_dir, "shaders", "basic.frag"),
        )

        # Parametry konfigurowalne
        self.size  = 3.0   # promien elipsoidy
        self.speed = 30.0  # jednostki/s
        self.mass  = 5.0   # masa (log-scale 1-10)

        # Stan
        self.state    = CometState.IDLE
        self.position = np.array([self.START_DIST, 5.0, 20.0], dtype=np.float32)
        self.direction = np.zeros(3, dtype=np.float32)
        self.impact_timer = 0.0
        self.impact_duration = 6.0
        self.tail_emit_timer = 0.0

        # Budowa siatki (elipsoida)
        self._rebuild_mesh()

        # System czasteczek (ogon + gruz uderzeniowy)
        self.particles = ParticleSystem(self.particle_shader)

        # System fragmentow orbitalnych
        self.orbital_debris = OrbitalDebris(self.basic_shader)

    def _rebuild_mesh(self):
        """Przebuduj siatke elipsoidy przy zmianie rozmiaru."""
        s = self.size
        # Elipsoida: troche spłaszczona jak asteroid
        verts, idxs = _make_ellipsoid_mesh(rx=s, ry=s*0.65, rz=s*0.80)
        self.index_count = len(idxs)

        if not hasattr(self, 'vao'):
            self.vao = gl.glGenVertexArrays(1)
            self.vbo = gl.glGenBuffers(1)
            self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, gl.GL_STATIC_DRAW)

        stride = 6 * verts.itemsize
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * verts.itemsize))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    def launch(self):
        """Aktywuj komete i wyslij w kierunku Ziemi."""
        if self.state != CometState.IDLE:
            return
        self._rebuild_mesh()
        # Staly start z lewej strony ekranu (ujemna os X), lekko z przodu (dodatnie Z)
        # Kamera patrzy z Z+ wiec kometa bedzie leciec z lewej strony widocznie
        self.position = np.array([
            -self.START_DIST,   # daleko z lewej strony
            8.0,                 # lekko powyzej centrum (widoczna trajektoria)
            30.0,                # blizej kamery (Z+), zeby lec w poprzek ekranu
        ], dtype=np.float32)
        # Kierunek: do centrum Ziemi (0,0,0) z malym losowym odchyleniem
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        diff = target - self.position
        diff /= np.linalg.norm(diff)
        self.direction = diff.astype(np.float32)
        self.state = CometState.FLYING
        self.particles.count = 0  # reset czastek

    def reset(self):
        """Przywroc stan idle i wyczys czasteczki."""
        self.state = CometState.IDLE
        self.impact_timer = 0.0
        self.particles.count = 0
        self.orbital_debris.reset()
        Earth.impact_progress = 0.0
        Earth.crater_radius   = 0.0

    def _emit_tail(self, dt: float):
        """Emituj czasteczki ogona podczas lotu."""
        self.tail_emit_timer += dt
        rate = 30  # czasteczki/s
        interval = 1.0 / rate
        while self.tail_emit_timer >= interval:
            self.tail_emit_timer -= interval
            # Ogon leci przeciwnie do kierunku ruchu
            tail_vel = (-self.direction * self.speed * 0.3
                        + np.random.randn(3).astype(np.float32) * 2.0)
            color = np.array([0.9, 0.6, 0.2], dtype=np.float32)
            self.particles.emit(
                pos      = self.position.copy(),
                vel      = tail_vel,
                color    = color,
                size     = self.size * 0.4,
                lifetime = 2.5,
                count    = 3,
            )

    def _emit_impact_explosion(self):
        """Emisja gruzu uderzeniowego."""
        N = 500
        for _ in range(N):
            # Losowy kierunek wybuchu
            phi   = np.random.uniform(0, math.pi)
            theta = np.random.uniform(0, 2*math.pi)
            vdir  = np.array([
                math.sin(phi)*math.cos(theta),
                abs(math.cos(phi)),           # glownie w gore
                math.sin(phi)*math.sin(theta),
            ], dtype=np.float32)

            # Predkosc zalezna od masy i predkosci komety
            spd = self.speed * (0.3 + np.random.rand() * 0.7) * (self.mass / 5.0) ** 0.3
            spd = min(spd, 80.0)

            color = np.array([1.0, 0.5, 0.1], dtype=np.float32)
            self.particles.emit(
                pos      = Earth.impact_point * (Comet.EARTH_RADIUS + 1.0),
                vel      = vdir * spd,
                color    = color,
                size     = self.size * 0.6,
                lifetime = 4.0,
                count    = 1,
            )

    def update(self, dt: float):
        if self.state == CometState.IDLE:
            return

        if self.state == CometState.FLYING:
            self.position += self.direction * self.speed * dt
            self._emit_tail(dt)

            # Sprawdz kolizje z Ziemia
            dist = np.linalg.norm(self.position)
            if dist <= self.EARTH_RADIUS + self.size * 0.8:
                self._start_impact()
            return

        if self.state == CometState.IMPACT:
            self.impact_timer += dt
            t = min(self.impact_timer / self.impact_duration, 1.0)
            Earth.impact_progress = math.sin(t * math.pi)  # wzrost i spadek

            # Rosnacy krater zalezy od masy i rozmiaru
            max_crater = min(0.8 + (self.mass / 10.0) * 0.6 + self.size / 20.0, 1.5)
            Earth.crater_radius = max_crater * min(t * 2.0, 1.0)

            # Emisja gruzu przez pierwsze 0.5s
            if self.impact_timer < 0.5:
                tail_vel = np.random.randn(3).astype(np.float32) * self.speed * 0.5
                color = np.array([1.0, 0.6, 0.1], dtype=np.float32)
                self.particles.emit(
                    pos      = Earth.impact_point * (Comet.EARTH_RADIUS + 0.5),
                    vel      = tail_vel,
                    color    = color,
                    size     = self.size * 0.8,
                    lifetime = 5.0,
                    count    = 8,
                )

            if self.impact_timer >= self.impact_duration:
                self.state = CometState.DONE

        self.particles.update(dt)
        self.orbital_debris.update(dt)

    def _start_impact(self):
        self.state = CometState.IMPACT
        self.impact_timer = 0.0
        # Punkt uderzenia = znormalizowana pozycja komety
        pt = self.position / (np.linalg.norm(self.position) + 1e-9)
        Earth.impact_point = pt.astype(np.float32)
        Earth.impact_progress = 0.0

        # Czas trwania animacji zalezny od masy
        self.impact_duration = 4.0 + self.mass * 0.4

        # Eksplozja gruzu
        self._emit_impact_explosion()

        # Fragmenty wchodza na orbite Ziemi
        self.orbital_debris.spawn(
            impact_point = Earth.impact_point,
            earth_radius = self.EARTH_RADIUS,
            comet_size   = self.size,
            comet_mass   = self.mass,
            comet_speed  = self.speed,
            comet_direction = self.direction,
        )

    def draw(self, view: np.ndarray, projection: np.ndarray, camera_pos: np.ndarray):
        # Rysuj czasteczki zawsze
        self.particles.update(0.0)  # tylko upload, update juz w update()
        self.particles.draw(view, projection)

        # Rysuj fragmenty orbitalne
        self.orbital_debris.draw(view, projection)

        # Rysuj cialo komety tylko gdy leci
        if self.state not in (CometState.FLYING,):
            return

        dist = np.linalg.norm(self.position)
        glow = max(0.0, 1.0 - (dist - self.EARTH_RADIUS) / 60.0)

        model = np.eye(4, dtype=np.float32)
        model[0, 3] = self.position[0]
        model[1, 3] = self.position[1]
        model[2, 3] = self.position[2]

        self.comet_shader.use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.comet_shader.program_id, "model"),
                               1, gl.GL_FALSE, model.T.copy())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.comet_shader.program_id, "view"),
                               1, gl.GL_FALSE, view.T.copy())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.comet_shader.program_id, "projection"),
                               1, gl.GL_FALSE, projection.T.copy())
        light_dir = np.array([1.0, 0.5, 0.8], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        gl.glUniform3fv(gl.glGetUniformLocation(self.comet_shader.program_id, "lightDir"), 1, light_dir)
        gl.glUniform3fv(gl.glGetUniformLocation(self.comet_shader.program_id, "viewPos"), 1,
                        camera_pos.astype(np.float32))
        gl.glUniform1f(gl.glGetUniformLocation(self.comet_shader.program_id, "cometGlow"), glow)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
