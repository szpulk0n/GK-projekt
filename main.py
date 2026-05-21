# main.py
import glfw
from OpenGL.GL import *
import numpy as np
import sys

from camera import Camera

# Ustawienia ekranu
SCR_WIDTH = 1280
SCR_HEIGHT = 720

# Kamera
camera = Camera(position=np.array([0.0, 5.0, 20.0], dtype=np.float32))
last_x = SCR_WIDTH / 2.0
last_y = SCR_HEIGHT / 2.0
first_mouse = True

# Timing
delta_time = 0.0
last_frame = 0.0

def process_input(window):
    global delta_time, camera
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera.process_keyboard("FORWARD", delta_time)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera.process_keyboard("BACKWARD", delta_time)
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera.process_keyboard("LEFT", delta_time)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera.process_keyboard("RIGHT", delta_time)

def mouse_callback(window, xpos, ypos):
    global last_x, last_y, first_mouse, camera
    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos # odwrócone, ponieważ oś Y rośnie w dół na ekranie

    last_x = xpos
    last_y = ypos

    camera.process_mouse_movement(xoffset, yoffset)

def framebuffer_size_callback(window, width, height):
    """Zapewnia poprawne skalowanie viewportu przy zmianie rozmiaru okna."""
    glViewport(0, 0, width, height)

def main():
    global delta_time, last_frame
    
    # Inicjalizacja biblioteki GLFW
    if not glfw.init():
        print("Błąd inicjalizacji GLFW")
        sys.exit(1)

    # Konfiguracja GLFW: Wymuszamy OpenGL 3.3 Core Profile (zabronione Fixed Pipeline)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE) # Potrzebne na macOS

    # Tworzenie okna
    window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "Interactive Cosmic Sandbox (FPS Camera)", None, None)
    if not window:
        print("Nie udało się utworzyć okna GLFW")
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    # Wyłączenie widoczności kursora i przechwycenie go przez okno (tryb FPS)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # Włączenie testowania głębokości (Depth Test) - niezbędne dla 3D
    glEnable(GL_DEPTH_TEST)

    # Inicjalizacja renderera (kompilacja shaderów, utworzenie buforów sfery)
    from renderer import SceneRenderer
    renderer = SceneRenderer()

    # Pętla główna (Render Loop)
    while not glfw.window_should_close(window):
        # 1. Zarządzanie czasem (Delta Time)
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        # 2. Przechwytywanie wejścia klawiatury (WASD + Escape)
        process_input(window)

        # 3. Renderowanie
        glClearColor(0.05, 0.05, 0.1, 1.0) # Głęboki kosmiczny kolor tła
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Pobranie macierzy z kamery
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix(SCR_WIDTH, SCR_HEIGHT)
        
        # Macierz modelu (próbna planeta w centrum)
        model_matrix = np.identity(4, dtype=np.float32)
        
        # Rysowanie planety testowej
        renderer.draw_sphere(view_matrix, projection_matrix, model_matrix)

        # 4. Zamiana buforów (Swap buffers) i obsługa zdarzeń GLFW
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Zakończenie pracy i zwolnienie zasobów GLFW
    glfw.terminate()

if __name__ == '__main__':
    main()
