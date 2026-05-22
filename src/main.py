import os
import glfw
import OpenGL.GL as gl
import numpy as np
import math

from window import Window
from shader import Shader
from camera import Camera
from physics import PhysicsEngine
from skybox import Skybox

# Scenariusz 4
from earth import Earth
from comet import Comet, CometState


def create_rotation_matrix(angle: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle)
    b = math.sin(angle)
    c = 1.0 - a
    x, y, z = axis

    return np.array([
        [a + x*x*c,   x*y*c - z*b, x*z*c + y*b, 0.0],
        [y*x*c + z*b, a + y*y*c,   y*z*c - x*b, 0.0],
        [z*x*c - y*b, z*y*c + x*b, a + z*z*c,   0.0],
        [0.0,         0.0,         0.0,         1.0]
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────
#  HUD scenariusza 4  (prosty rasteryzator znaków)
# ─────────────────────────────────────────────────────────

# Minimalna czcionka 5×7 pikseli (32-127 ASCII) zakodowana jako bitmapa
# Ponizej uzywamy glWindowPos + glBitmap (staly potok) albo po prostu
# rysujemy quady z OpenGL 3.3. Wybieramy podejscie "colored quads" zeby
# nie lamac Core Profile.

HUD_VERT_SRC = """
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec2 uResolution;
void main(){
    vec2 ndc = (aPos / uResolution) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

HUD_FRAG_SRC = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main(){ FragColor = uColor; }
"""


class HUDRenderer:
    """Rysuje paski (quady) HUD w przestrzeni ekranu."""

    def __init__(self):
        # Kompiluj shadery inline
        def _compile(src, stype):
            s = gl.glCreateShader(stype)
            gl.glShaderSource(s, src)
            gl.glCompileShader(s)
            if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(s))
            return s

        vs = _compile(HUD_VERT_SRC, gl.GL_VERTEX_SHADER)
        fs = _compile(HUD_FRAG_SRC, gl.GL_FRAGMENT_SHADER)
        self.prog = gl.glCreateProgram()
        gl.glAttachShader(self.prog, vs)
        gl.glAttachShader(self.prog, fs)
        gl.glLinkProgram(self.prog)
        gl.glDeleteShader(vs); gl.glDeleteShader(fs)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 48, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

    def draw_rect(self, x, y, w, h, r, g, b, a=1.0, res=(800, 600)):
        """Rysuje wypelniony prostokat w pikselach (od lewego dolnego rogu)."""
        verts = np.array([
            x,   y,
            x+w, y,
            x+w, y+h,
            x,   y,
            x+w, y+h,
            x,   y+h,
        ], dtype=np.float32)

        gl.glUseProgram(self.prog)
        gl.glUniform2f(gl.glGetUniformLocation(self.prog, "uResolution"), res[0], res[1])
        gl.glUniform4f(gl.glGetUniformLocation(self.prog, "uColor"), r, g, b, a)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)


# ─────────────────────────────────────────────────────────
#  Główna funkcja
# ─────────────────────────────────────────────────────────

def main() -> None:
    width, height = 800, 600
    window = Window(width, height, "GK Projekt - Scenariusze 1-4")

    glfw.set_input_mode(window.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    gl.glEnable(gl.GL_DEPTH_TEST)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vert_path = os.path.join(base_dir, "shaders", "basic.vert")
    frag_path = os.path.join(base_dir, "shaders", "basic.frag")
    shader = Shader(vert_path, frag_path)

    camera = Camera(position=(0.0, 20.0, 60.0))

    last_x, last_y = width / 2.0, height / 2.0
    first_mouse = True

    def mouse_callback(win, xpos, ypos):
        nonlocal last_x, last_y, first_mouse
        if first_mouse:
            last_x = xpos; last_y = ypos; first_mouse = False
        xoffset = xpos - last_x
        yoffset = last_y - ypos
        last_x = xpos; last_y = ypos
        camera.process_mouse_movement(xoffset, yoffset)

    glfw.set_cursor_pos_callback(window.window, mouse_callback)

    time_scale      = 1.0
    current_scenario = 1
    N_PARTICLES      = 10000

    # ── Scenariusz 4 – stan ──────────────────────────────
    earth  = None
    comet  = None
    hud    = None

    # Suwaki: 0=rozmiar 1=predkosc 2=masa
    SLIDER_NAMES  = ["Wielkosc komety", "Predkosc komety", "Masa komety"]
    SLIDER_MIN    = [1.0,   5.0,  1.0]
    SLIDER_MAX    = [10.0, 80.0, 10.0]
    SLIDER_DEF    = [3.0,  30.0,  5.0]
    slider_vals   = list(SLIDER_DEF)
    active_slider = 0        # aktywny suwak (Shift przełącza)
    sc4_ready     = False    # czy scenariusz 4 jest zainicjowany
    # ────────────────────────────────────────────────────

    def process_input(win, delta_time):
        nonlocal time_scale
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
            camera.process_keyboard("FORWARD", delta_time)
        if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
            camera.process_keyboard("BACKWARD", delta_time)
        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
            camera.process_keyboard("LEFT", delta_time)
        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
            camera.process_keyboard("RIGHT", delta_time)
        if glfw.get_key(win, glfw.KEY_UP) == glfw.PRESS:
            time_scale = min(time_scale + 0.5 * delta_time, 5.0)
        if glfw.get_key(win, glfw.KEY_DOWN) == glfw.PRESS:
            time_scale = max(time_scale - 0.5 * delta_time, 0.0)

    # ── Cube vertices (scenariusze 1-3) ─────────────────
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

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, gl.ctypes.c_void_p(3*vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    physics = PhysicsEngine(N_PARTICLES, scenario=current_scenario)
    instance_positions = physics.positions.copy()

    instance_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions, gl.GL_DYNAMIC_DRAW)
    gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 3*instance_positions.itemsize, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribDivisor(2, 1)
    gl.glBindVertexArray(0)

    # ── Cooldown dla klawiszy (zapobiega wielokrotnej rejestracji) ─
    shift_cooldown   = 0.0
    right_cooldown   = 0.0
    left_cooldown    = 0.0
    enter_cooldown   = 0.0
    r_cooldown       = 0.0

    def key_callback(win, key, scancode, action, mods):
        nonlocal current_scenario, N_PARTICLES, physics, instance_vbo, instance_positions
        nonlocal earth, comet, hud, sc4_ready, slider_vals, active_slider

        if action == glfw.PRESS:
            new_scenario = current_scenario
            if key == glfw.KEY_1: new_scenario = 1
            elif key == glfw.KEY_2: new_scenario = 2
            elif key == glfw.KEY_3: new_scenario = 3
            elif key == glfw.KEY_4: new_scenario = 4

            if new_scenario != current_scenario:
                current_scenario = new_scenario

                if new_scenario == 4:
                    # Inicjalizuj scenariusz 4
                    if not sc4_ready:
                        earth = Earth(radius=10.0)
                        comet = Comet()
                        hud   = HUDRenderer()
                        sc4_ready = True
                    else:
                        # Reset komety
                        comet.reset()
                        slider_vals = list(SLIDER_DEF)
                        active_slider = 0
                    camera.position = np.array([0.0, 10.0, 40.0], dtype=np.float32)
                    camera.yaw = -90.0
                    camera.pitch = -10.0
                    camera._update_camera_vectors()
                else:
                    N_PARTICLES = 10000
                    physics = PhysicsEngine(N_PARTICLES, center_mass=10000.0, scenario=current_scenario)
                    instance_positions = physics.positions.copy()
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions, gl.GL_DYNAMIC_DRAW)

    glfw.set_key_callback(window.window, key_callback)

    skybox = Skybox()

    # ── Główna pętla ─────────────────────────────────────
    while window.is_running():
        delta = window.update()
        process_input(window.window, delta)

        # Cooldowny
        shift_cooldown  = max(0.0, shift_cooldown  - delta)
        right_cooldown  = max(0.0, right_cooldown  - delta)
        left_cooldown   = max(0.0, left_cooldown   - delta)
        enter_cooldown  = max(0.0, enter_cooldown  - delta)
        r_cooldown      = max(0.0, r_cooldown      - delta)

        # ── Aktualizacja sceny ───────────────────────────
        sim_delta = min(delta, 0.05) * time_scale

        if current_scenario in (1, 2, 3):
            if sim_delta > 0.0:
                new_positions = physics.update(sim_delta)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)

        elif current_scenario == 4 and sc4_ready:
            # Klawiatura scenariusza 4
            win = window.window

            # SHIFT – zmiana aktywnego suwaka
            if (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS):
                if shift_cooldown <= 0.0:
                    active_slider = (active_slider + 1) % 3
                    shift_cooldown = 0.3

            # Strzałki – wartość suwaka
            step = [(SLIDER_MAX[i] - SLIDER_MIN[i]) / 40.0 for i in range(3)]
            if glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS and right_cooldown <= 0.0:
                slider_vals[active_slider] = min(
                    slider_vals[active_slider] + step[active_slider], SLIDER_MAX[active_slider])
                right_cooldown = 0.05
            if glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS and left_cooldown <= 0.0:
                slider_vals[active_slider] = max(
                    slider_vals[active_slider] - step[active_slider], SLIDER_MIN[active_slider])
                left_cooldown = 0.05

            # ENTER – uruchom kometę
            if (glfw.get_key(win, glfw.KEY_ENTER) == glfw.PRESS and
                    enter_cooldown <= 0.0 and
                    comet.state in (CometState.IDLE, CometState.DONE)):
                comet.size  = slider_vals[0]
                comet.speed = slider_vals[1]
                comet.mass  = slider_vals[2]
                comet.reset()
                comet.size  = slider_vals[0]
                comet.speed = slider_vals[1]
                comet.mass  = slider_vals[2]
                comet.launch()
                enter_cooldown = 0.5

            # R – reset
            if glfw.get_key(win, glfw.KEY_R) == glfw.PRESS and r_cooldown <= 0.0:
                comet.reset()
                slider_vals = list(SLIDER_DEF)
                active_slider = 0
                r_cooldown = 0.5

            if sim_delta > 0.0:
                earth.update(sim_delta)
                comet.update(sim_delta)

        # ── Renderowanie ─────────────────────────────────
        gl.glClearColor(0.02, 0.02, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        cw, ch = glfw.get_window_size(window.window)
        if ch == 0: ch = 1
        aspect = cw / ch

        view       = camera.get_view_matrix()
        projection = camera.get_projection_matrix(aspect, near=0.1, far=500.0)

        if current_scenario in (1, 2, 3):
            shader.use()
            shader.set_mat4("view",       view.T.copy())
            shader.set_mat4("projection", projection.T.copy())
            gl.glBindVertexArray(vao)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, N_PARTICLES)

        elif current_scenario == 4 and sc4_ready:
            cam_pos = camera.position
            earth.draw(view, projection, cam_pos)
            comet.draw(view, projection, cam_pos)

            # ── HUD ──────────────────────────────────────
            _draw_hud4(hud, slider_vals, active_slider, comet, (cw, ch),
                       SLIDER_NAMES, SLIDER_MIN, SLIDER_MAX)

        skybox.draw(view, projection)

    window.terminate()


# ─────────────────────────────────────────────────────────
#  Rysowanie HUD scenariusza 4
# ─────────────────────────────────────────────────────────

def _draw_hud4(hud: 'HUDRenderer', vals, active, comet: 'Comet',
               res, names, mins, maxs):
    """Rysuje panel sterowania suwakami i status komety."""
    if hud is None:
        return

    w, h = res

    # ── Kolory i ikony dla każdego suwaka ──────────────
    SLIDER_COLORS = [
        (0.25, 0.65, 1.00),   # niebieski  – rozmiar
        (1.00, 0.45, 0.05),   # pomarancz  – predkosc
        (0.75, 0.20, 0.90),   # fioletowy  – masa
    ]
    SLIDER_UNITS = [
        "promien [1-10 j]",
        "predkosc [5-80 j/s]",
        "masa [1-10 skala]",
    ]

    # ── Wymiary panelu ─────────────────────────────────
    panel_x = 18
    panel_w = 380
    ROW_H   = 58      # wysokosc jednego wiersza suwaka
    HEADER  = 30
    panel_h = HEADER + 3 * ROW_H + 10
    panel_y = h - panel_h - 18

    # Cien / ramka zewnetrzna
    hud.draw_rect(panel_x - 3, panel_y - 3, panel_w + 6, panel_h + 6,
                  0.0, 0.0, 0.0, 0.70, res)

    # Tlo panelu
    hud.draw_rect(panel_x, panel_y, panel_w, panel_h,
                  0.04, 0.06, 0.10, 0.88, res)

    # Naglowek
    hud.draw_rect(panel_x, panel_y + panel_h - HEADER, panel_w, HEADER,
                  0.08, 0.16, 0.28, 0.95, res)
    # Cienka linia pod naglowkiem
    hud.draw_rect(panel_x, panel_y + panel_h - HEADER, panel_w, 2,
                  0.20, 0.40, 0.70, 1.0, res)

    # ── Suwaki ─────────────────────────────────────────
    for i in range(3):
        row_y   = panel_y + panel_h - HEADER - (i + 1) * ROW_H
        is_active = (i == active)
        rc, gc, bc = SLIDER_COLORS[i]

        # Podswietlenie aktywnego wiersza
        if is_active:
            hud.draw_rect(panel_x + 1, row_y + 1, panel_w - 2, ROW_H - 2,
                          rc * 0.12, gc * 0.12, bc * 0.12, 0.80, res)
            # Lewa kolorowa kreska aktywnego
            hud.draw_rect(panel_x, row_y, 4, ROW_H, rc, gc, bc, 1.0, res)
            # Gorny / dolny border
            hud.draw_rect(panel_x, row_y + ROW_H - 2, panel_w, 2, rc, gc, bc, 0.6, res)
            hud.draw_rect(panel_x, row_y,              panel_w, 2, rc, gc, bc, 0.6, res)
        else:
            hud.draw_rect(panel_x, row_y, 2, ROW_H, rc * 0.5, gc * 0.5, bc * 0.5, 0.7, res)

        # --- Pasek (tło + wypełnienie + uchwyt) ---
        bar_x = panel_x + 14
        bar_y = row_y + 8
        bar_w = panel_w - 28
        bar_h = 18    # grubszy pasek

        # Cien pod paskiem
        hud.draw_rect(bar_x + 2, bar_y - 2, bar_w, bar_h,
                      0.0, 0.0, 0.0, 0.5, res)
        # Tlo paska – ciemna szarosc
        hud.draw_rect(bar_x, bar_y, bar_w, bar_h,
                      0.12, 0.12, 0.14, 1.0, res)

        # Wypelnienie
        t = (vals[i] - mins[i]) / (maxs[i] - mins[i])
        fill_w = max(4, int(bar_w * t))

        # Gradient: ciemniejszy start → jasniejszy koniec (2-etapowy przez quady)
        half = fill_w // 2
        hud.draw_rect(bar_x, bar_y, half, bar_h,
                      rc * 0.55, gc * 0.55, bc * 0.55, 1.0, res)
        hud.draw_rect(bar_x + half, bar_y, fill_w - half, bar_h,
                      rc, gc, bc, 1.0, res)

        # Połysk na górze paska
        hud.draw_rect(bar_x, bar_y + bar_h - 4, fill_w, 4,
                      min(rc + 0.3, 1.0), min(gc + 0.3, 1.0), min(bc + 0.3, 1.0), 0.35, res)

        # Uchwyt (bialy pionowy pasek)
        hx = bar_x + fill_w - 4
        hud.draw_rect(hx, bar_y - 3, 8, bar_h + 6, 1.0, 1.0, 1.0, 0.95, res)
        hud.draw_rect(hx + 2, bar_y - 3, 4, bar_h + 6, rc, gc, bc, 0.6, res)

        # Skala minimalna i maksymalna (krotkie kreski na koncach)
        hud.draw_rect(bar_x, bar_y - 4, 2, 4, 0.5, 0.5, 0.5, 0.8, res)
        hud.draw_rect(bar_x + bar_w - 2, bar_y - 4, 2, 4, 0.5, 0.5, 0.5, 0.8, res)

        # --- Etykieta suwaka z jasnym tlem i podpisem ---
        label_y = bar_y + bar_h + 4
        label_h = 20

        # Jasne polprzezroczyste tlo pod calym wierszem etykiety (widoczne na ciemnym tle kosmosu)
        hud.draw_rect(bar_x - 2, label_y - 2, bar_w + 4, label_h + 4,
                      0.88, 0.90, 0.95, 0.22, res)
        # Gorna linia koloru suwaka (obrys etykiety)
        hud.draw_rect(bar_x - 2, label_y - 2, bar_w + 4, 2,
                      rc, gc, bc, 0.6, res)
        # Dolna linia
        hud.draw_rect(bar_x - 2, label_y + label_h + 2, bar_w + 4, 1,
                      rc, gc, bc, 0.3, res)

        # Kolorowy kwadracik – ikona (identyfikator suwaka)
        hud.draw_rect(bar_x, label_y + 5, 12, 12, rc, gc, bc, 1.0, res)
        hud.draw_rect(bar_x + 2, label_y + 7, 8, 8, 1.0, 1.0, 1.0, 0.35, res)

        # Pasek wartosci numerycznej – mini wizualizacja wartosci
        num_bar_x = bar_x + 18
        num_bar_total = 80
        num_bar_w = int(num_bar_total * t)
        # Tlo
        hud.draw_rect(num_bar_x, label_y + 7, num_bar_total, 8, 0.10, 0.10, 0.12, 0.88, res)
        # Wypelnienie
        hud.draw_rect(num_bar_x, label_y + 7, num_bar_w, 8, rc * 0.9, gc * 0.9, bc * 0.9, 1.0, res)
        # Polysek
        hud.draw_rect(num_bar_x, label_y + 7, num_bar_w, 3, 1.0, 1.0, 1.0, 0.25, res)
        # Kursor wartosci
        if num_bar_w > 2:
            hud.draw_rect(num_bar_x + num_bar_w - 2, label_y + 5, 3, 12,
                          1.0, 1.0, 1.0, 0.95, res)

        # Etykieta jednostki – jasne tlo, widoczne nawet na tle kosmosu
        unit_bg_x = num_bar_x + num_bar_total + 8
        unit_bg_w = bar_w - (unit_bg_x - bar_x) - 2
        if unit_bg_w > 30:
            # Jasniejsze tlo pod etykieta jednostki
            hud.draw_rect(unit_bg_x, label_y + 3, unit_bg_w, label_h - 6,
                          0.92, 0.94, 0.98, 0.30, res)
            # Lewa krawedz kolorowa
            hud.draw_rect(unit_bg_x, label_y + 3, 3, label_h - 6, rc, gc, bc, 0.85, res)
            # Srodkowa linia pozioma jako skala wartosci min-max
            scale_y = label_y + 12
            hud.draw_rect(unit_bg_x + 6, scale_y, unit_bg_w - 12, 2, 0.55, 0.55, 0.60, 0.7, res)
            # Znacznik min (lewy)
            hud.draw_rect(unit_bg_x + 6, scale_y - 3, 2, 8, 0.65, 0.65, 0.70, 0.8, res)
            # Znacznik max (prawy)
            hud.draw_rect(unit_bg_x + unit_bg_w - 8, scale_y - 3, 2, 8, 0.65, 0.65, 0.70, 0.8, res)
            # Aktualny znacznik wartosci (kolorowy, wyrazny)
            cur_x = unit_bg_x + 6 + int((unit_bg_w - 14) * t)
            hud.draw_rect(cur_x - 1, scale_y - 5, 4, 12, rc, gc, bc, 1.0, res)
            hud.draw_rect(cur_x, scale_y - 5, 2, 12, 1.0, 1.0, 1.0, 0.5, res)

    # ── Separator ──────────────────────────────────────
    sep_y = panel_y - 8
    hud.draw_rect(panel_x, sep_y, panel_w, 1, 0.3, 0.3, 0.3, 0.5, res)

    # ── Status komety ──────────────────────────────────
    STATUS = {
        'idle':   ((0.40, 0.40, 0.40), (0.10, 0.10, 0.10)),
        'flying': ((0.20, 0.85, 0.25), (0.03, 0.15, 0.04)),
        'impact': ((1.00, 0.30, 0.00), (0.18, 0.04, 0.00)),
        'done':   ((0.55, 0.55, 0.60), (0.08, 0.08, 0.10)),
    }
    (r2, g2, b2), (rb, gb, bb) = STATUS.get(comet.state, ((1,1,1), (0.1,0.1,0.1)))

    status_h = 38
    status_y = panel_y - status_h - 12

    # Cien
    hud.draw_rect(panel_x - 3, status_y - 3, panel_w + 6, status_h + 6,
                  0.0, 0.0, 0.0, 0.55, res)
    # Tlo
    hud.draw_rect(panel_x, status_y, panel_w, status_h,
                  rb, gb, bb, 0.92, res)
    # Lewa krawedz kolorowa
    hud.draw_rect(panel_x, status_y, 6, status_h, r2, g2, b2, 1.0, res)
    # Gorny pasek koloru
    hud.draw_rect(panel_x, status_y + status_h - 3, panel_w, 3, r2, g2, b2, 0.8, res)

    # Pasek postępu uderzenia
    if comet.state == 'impact':
        t_imp = min(comet.impact_timer / comet.impact_duration, 1.0)
        hud.draw_rect(panel_x + 10, status_y + 8,  panel_w - 20, 14,
                      0.15, 0.04, 0.01, 0.9, res)
        imp_w = int((panel_w - 20) * t_imp)
        hud.draw_rect(panel_x + 10, status_y + 8, imp_w, 14,
                      1.0, 0.40, 0.0, 1.0, res)
        hud.draw_rect(panel_x + 10, status_y + 18, imp_w, 4,
                      1.0, 0.85, 0.3, 0.5, res)

    # ── Legenda klawiszy (dół ekranu) ─────────────────
    keys_y = 0
    keys_h = 28
    hud.draw_rect(0, keys_y, w, keys_h, 0.02, 0.02, 0.04, 0.80, res)
    hud.draw_rect(0, keys_y + keys_h - 2, w, 2, 0.15, 0.25, 0.40, 0.7, res)

    # Kolorowe "przyciski" klawiszy
    def key_btn(kx, ky, kw, r, g, b):
        hud.draw_rect(kx, ky + 4, kw, 18, r*0.3, g*0.3, b*0.3, 0.9, res)
        hud.draw_rect(kx, ky + 4, kw, 3,  r,     g,     b,     1.0, res)
        hud.draw_rect(kx, ky + 19, kw, 3, r*0.5, g*0.5, b*0.5, 0.8, res)

    key_btn(10,  keys_y, 55, 0.4, 0.8, 1.0)   # ← →
    key_btn(75,  keys_y, 50, 0.4, 0.8, 1.0)   # SHIFT
    key_btn(135, keys_y, 50, 0.2, 0.9, 0.3)   # ENTER
    key_btn(195, keys_y, 30, 0.9, 0.3, 0.2)   # R
    key_btn(235, keys_y, 70, 0.6, 0.6, 0.6)   # WASD/mysz


if __name__ == "__main__":
    main()
