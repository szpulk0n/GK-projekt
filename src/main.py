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

def main() -> None:
    width, height = 800, 600
    window = Window(width, height, "Phase 2 - Free-Fly Camera")
    
    # Hide and capture cursor
    glfw.set_input_mode(window.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    gl.glEnable(gl.GL_DEPTH_TEST)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vert_path = os.path.join(base_dir, "shaders", "basic.vert")
    frag_path = os.path.join(base_dir, "shaders", "basic.frag")
    shader = Shader(vert_path, frag_path)
    
    camera = Camera(position=(0.0, 20.0, 60.0))
    
    # Mouse state
    last_x, last_y = width / 2.0, height / 2.0
    first_mouse = True

    def mouse_callback(win, xpos, ypos):
        nonlocal last_x, last_y, first_mouse
        
        if first_mouse:
            last_x = xpos
            last_y = ypos
            first_mouse = False
            
        xoffset = xpos - last_x
        yoffset = last_y - ypos # reversed since y-coordinates go from bottom to top
        
        last_x = xpos
        last_y = ypos
        
        camera.process_mouse_movement(xoffset, yoffset)

    glfw.set_cursor_pos_callback(window.window, mouse_callback)

    time_scale = 1.0
    current_scenario = 1
    N_PARTICLES = 10000

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
            
        # Time scaling
        if glfw.get_key(win, glfw.KEY_UP) == glfw.PRESS:
            time_scale += 0.5 * delta_time
        if glfw.get_key(win, glfw.KEY_DOWN) == glfw.PRESS:
            time_scale -= 0.5 * delta_time
            if time_scale < 0.0:
                time_scale = 0.0

    # Cube vertices (position: 3 floats, color: 3 floats)
    vertices = np.array([
        # Back face
        -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,
         0.5, -0.5, -0.5,  0.0, 1.0, 0.0,
         0.5,  0.5, -0.5,  0.0, 0.0, 1.0,
         0.5,  0.5, -0.5,  0.0, 0.0, 1.0,
        -0.5,  0.5, -0.5,  1.0, 1.0, 0.0,
        -0.5, -0.5, -0.5,  1.0, 0.0, 0.0,

        # Front face
        -0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
         0.5, -0.5,  0.5,  0.0, 1.0, 0.0,
         0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
         0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,  1.0, 1.0, 0.0,
        -0.5, -0.5,  0.5,  1.0, 0.0, 0.0,

        # Left face
        -0.5,  0.5,  0.5,  1.0, 0.0, 1.0,
        -0.5,  0.5, -0.5,  0.0, 1.0, 1.0,
        -0.5, -0.5, -0.5,  1.0, 1.0, 1.0,
        -0.5, -0.5, -0.5,  1.0, 1.0, 1.0,
        -0.5, -0.5,  0.5,  0.0, 0.0, 0.0,
        -0.5,  0.5,  0.5,  1.0, 0.0, 1.0,

        # Right face
         0.5,  0.5,  0.5,  1.0, 0.0, 1.0,
         0.5,  0.5, -0.5,  0.0, 1.0, 1.0,
         0.5, -0.5, -0.5,  1.0, 1.0, 1.0,
         0.5, -0.5, -0.5,  1.0, 1.0, 1.0,
         0.5, -0.5,  0.5,  0.0, 0.0, 0.0,
         0.5,  0.5,  0.5,  1.0, 0.0, 1.0,

        # Bottom face
        -0.5, -0.5, -0.5,  0.5, 0.5, 0.5,
         0.5, -0.5, -0.5,  0.5, 0.0, 0.0,
         0.5, -0.5,  0.5,  0.0, 0.5, 0.0,
         0.5, -0.5,  0.5,  0.0, 0.5, 0.0,
        -0.5, -0.5,  0.5,  0.0, 0.0, 0.5,
        -0.5, -0.5, -0.5,  0.5, 0.5, 0.5,

        # Top face
        -0.5,  0.5, -0.5,  0.5, 0.5, 0.5,
         0.5,  0.5, -0.5,  0.5, 0.0, 0.0,
         0.5,  0.5,  0.5,  0.0, 0.5, 0.0,
         0.5,  0.5,  0.5,  0.0, 0.5, 0.0,
        -0.5,  0.5,  0.5,  0.0, 0.0, 0.5,
        -0.5,  0.5, -0.5,  0.5, 0.5, 0.5,
    ], dtype=np.float32)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)
    
    # Static geometry (cube vertices & colors)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(0)
    
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * vertices.itemsize, gl.ctypes.c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    # --- INITIAL SCENARIO SETUP ---
    physics = PhysicsEngine(N_PARTICLES, scenario=current_scenario)
    instance_positions = physics.positions.copy()

    instance_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions, gl.GL_DYNAMIC_DRAW)

    # Attribute 2: Instance Offset (vec3)
    gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * instance_positions.itemsize, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribDivisor(2, 1) 

    gl.glBindVertexArray(0)
    
    # --- KEY CALLBACK FOR SCENARIOS ---
    def key_callback(win, key, scancode, action, mods):
        nonlocal current_scenario, N_PARTICLES, physics, instance_vbo, instance_positions
        if action == glfw.PRESS:
            new_scenario = current_scenario
            if key == glfw.KEY_1:
                new_scenario = 1
            elif key == glfw.KEY_2:
                new_scenario = 2
            elif key == glfw.KEY_3:
                new_scenario = 3
                
            if new_scenario != current_scenario:
                current_scenario = new_scenario
                N_PARTICLES = 10000
                physics = PhysicsEngine(N_PARTICLES, center_mass=10000.0, scenario=current_scenario)
                instance_positions = physics.positions.copy()
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions, gl.GL_DYNAMIC_DRAW)

    glfw.set_key_callback(window.window, key_callback)

    # --- SKYBOX SETUP ---
    skybox = Skybox()

    while window.is_running():
        delta = window.update()
        
        process_input(window.window, delta)
        
        # --- PHYSICS UPDATE ---
        # Apply time scale
        sim_delta = min(delta, 0.05) * time_scale
        
        # Only update if time is moving
        if sim_delta > 0.0:
            new_positions = physics.update(sim_delta)
            
            # Update GPU VBO with new positions
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instance_vbo)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)
        
        gl.glClearColor(0.05, 0.05, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        current_width, current_height = glfw.get_window_size(window.window)
        if current_height == 0: current_height = 1
        aspect = current_width / current_height
        
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(aspect)
        
        # Draw Particles
        shader.use()
        shader.set_mat4("view", view.T.copy())
        shader.set_mat4("projection", projection.T.copy())

        gl.glBindVertexArray(vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, N_PARTICLES)
        
        # Draw Skybox last for depth optimization
        skybox.draw(view, projection)
        
    window.terminate()

if __name__ == "__main__":
    main()
