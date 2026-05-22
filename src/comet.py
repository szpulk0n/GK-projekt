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
#  Glowna klasa komety
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

        # System czasteczek
        self.particles = ParticleSystem(self.particle_shader)

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

    def draw(self, view: np.ndarray, projection: np.ndarray, camera_pos: np.ndarray):
        # Rysuj czasteczki zawsze
        self.particles.update(0.0)  # tylko upload, update juz w update()
        self.particles.draw(view, projection)

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
