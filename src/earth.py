import OpenGL.GL as gl
import numpy as np
import math
import os
from shader import Shader


def _make_sphere_mesh(stacks: int = 64, slices: int = 64):
    """
    Tworzy siatke sfery z trojkatow (UV-sphere).
    Kazdy czworobok jest podzielony na 2 trojkaty, co daje
    ladny efekt zakrzywien przy renderowaniu bez tekstur.
    Zwraca (vertices, indices) gdzie vertices ma format:
        [x, y, z, nx, ny, nz, u, v]
    """
    vertices = []
    indices = []

    for i in range(stacks + 1):
        phi = math.pi * i / stacks          # od 0 do pi (gore -> dol)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        v = i / stacks                       # tex coord V

        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices  # od 0 do 2pi
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            u = j / slices                       # tex coord U

            # Pozycja na sferze jednostkowej
            x = sin_phi * cos_theta
            y = cos_phi
            z = sin_phi * sin_theta

            # Normalna = pozycja (sfera jednostkowa)
            vertices.extend([x, y, z, x, y, z, u, v])

    # Indeksy trojkatow
    for i in range(stacks):
        for j in range(slices):
            # Dwa trojkaty na kazdy czworobok
            top_left     = i * (slices + 1) + j
            top_right    = top_left + 1
            bottom_left  = (i + 1) * (slices + 1) + j
            bottom_right = bottom_left + 1

            # Trojkat 1
            indices.extend([top_left, bottom_left, top_right])
            # Trojkat 2
            indices.extend([top_right, bottom_left, bottom_right])

    verts = np.array(vertices, dtype=np.float32)
    idxs  = np.array(indices,  dtype=np.uint32)
    return verts, idxs


class Earth:
    """Renderuje kule ziemska jako sfere z proceduralna tekstura."""

    # Stan uderzenia (wspoldzielony globalnie w scenariuszu 4)
    impact_progress: float = 0.0   # 0 = brak, 1 = szczyt
    impact_point:    np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    crater_radius:   float = 0.0
    radius:          float = 10.0

    def __init__(self, radius: float = 10.0, stacks: int = 64, slices: int = 64):
        Earth.radius = radius
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "earth.vert"),
            os.path.join(base_dir, "shaders", "earth.frag"),
        )

        vertices, indices = _make_sphere_mesh(stacks, slices)
        self.index_count = len(indices)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        stride = 8 * vertices.itemsize
        # aPos (location=0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # aNormal (location=1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # aTexCoord (location=2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        # Macierz modelu - skala i pozycja
        self._model = np.eye(4, dtype=np.float32)
        self._model[0, 0] = radius
        self._model[1, 1] = radius
        self._model[2, 2] = radius

        self._rotation_y = 0.0

    def update(self, delta_time: float):
        """Powolna rotacja Ziemi."""
        self._rotation_y += delta_time * 0.8  # Umiarkowanie spowolniona rotacja Ziemi

    def _build_model_matrix(self) -> np.ndarray:
        """Macierz modelu = rotacja Y * skala."""
        c = math.cos(self._rotation_y)
        s = math.sin(self._rotation_y)
        rot = np.array([
            [ c,  0,  s,  0],
            [ 0,  1,  0,  0],
            [-s,  0,  c,  0],
            [ 0,  0,  0,  1],
        ], dtype=np.float32)
        scale = np.diag([Earth.radius, Earth.radius, Earth.radius, 1.0]).astype(np.float32)
        return rot @ scale

    def draw(self, view: np.ndarray, projection: np.ndarray, camera_pos: np.ndarray):
        model = self._build_model_matrix()

        self.shader.use()

        loc_model = gl.glGetUniformLocation(self.shader.program_id, "model")
        gl.glUniformMatrix4fv(loc_model, 1, gl.GL_FALSE, model.T.copy())

        loc_view = gl.glGetUniformLocation(self.shader.program_id, "view")
        gl.glUniformMatrix4fv(loc_view, 1, gl.GL_FALSE, view.T.copy())

        loc_proj = gl.glGetUniformLocation(self.shader.program_id, "projection")
        gl.glUniformMatrix4fv(loc_proj, 1, gl.GL_FALSE, projection.T.copy())

        # Swiatlo - sloneczne z boku
        light_dir = np.array([1.0, 0.5, 0.8], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        loc_light = gl.glGetUniformLocation(self.shader.program_id, "lightDir")
        gl.glUniform3fv(loc_light, 1, light_dir)

        loc_vp = gl.glGetUniformLocation(self.shader.program_id, "viewPos")
        gl.glUniform3fv(loc_vp, 1, camera_pos.astype(np.float32))

        # Uderzenie
        loc_imp = gl.glGetUniformLocation(self.shader.program_id, "impactProgress")
        gl.glUniform1f(loc_imp, Earth.impact_progress)

        loc_ipt = gl.glGetUniformLocation(self.shader.program_id, "impactPoint")
        gl.glUniform3fv(loc_ipt, 1, Earth.impact_point)

        loc_cr = gl.glGetUniformLocation(self.shader.program_id, "craterRadius")
        gl.glUniform1f(loc_cr, Earth.crater_radius)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
