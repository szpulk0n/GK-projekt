# renderer.py
import ctypes
import math
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as gl_shaders

from shaders import BASIC_VERTEX_SHADER, BASIC_FRAGMENT_SHADER

def generate_sphere(radius=1.0, sectors=36, stacks=18):
    """
    Ręcznie generuje wierzchołki (pozycje + kolory) i indeksy dla sfery geometrycznej.
    Zwraca tuple: (vertices_array, indices_array) typu float32 i uint32.
    """
    vertices = []
    indices = []
    
    # Generowanie wierzchołków
    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * math.pi / stacks
        xy = radius * math.cos(stack_angle)
        z = radius * math.sin(stack_angle)
        
        for j in range(sectors + 1):
            sector_angle = j * 2 * math.pi / sectors
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            
            # Traktujemy wektor normalny (kierunek od środka do wierzchołka) jako kolor RGB (wartości absolutne)
            nx = x / radius
            ny = y / radius
            nz = z / radius
            r, g, b = abs(nx), abs(ny), abs(nz)
            
            # Format: (x, y, z, r, g, b)
            vertices.extend([x, y, z, r, g, b])
            
    # Generowanie indeksów (triangulacja za pomocą EBO)
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1
        for j in range(sectors):
            if i != 0:
                indices.extend([k1, k2, k1 + 1])
            if i != (stacks - 1):
                indices.extend([k1 + 1, k2, k2 + 1])
            k1 += 1
            k2 += 1
            
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


class SceneRenderer:
    def __init__(self):
        # 1. Kompilacja shaderów
        self.shader_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(BASIC_VERTEX_SHADER, GL_VERTEX_SHADER),
            gl_shaders.compileShader(BASIC_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # 2. Generowanie geometrii planety (sfery)
        self.sphere_vertices, self.sphere_indices = generate_sphere(1.0, 36, 18)
        self.index_count = len(self.sphere_indices)
        
        # 3. Inicjalizacja VAO, VBO, EBO (Core Profile)
        self.sphere_vao = glGenVertexArrays(1)
        self.sphere_vbo = glGenBuffers(1)
        self.sphere_ebo = glGenBuffers(1)
        
        # Bindowanie VAO
        glBindVertexArray(self.sphere_vao)
        
        # Bindowanie VBO i ładowanie wierzchołków do pamięci GPU
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.sphere_vertices.nbytes, self.sphere_vertices, GL_STATIC_DRAW)
        
        # Bindowanie EBO i ładowanie indeksów
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.sphere_indices.nbytes, self.sphere_indices, GL_STATIC_DRAW)
        
        # Definiowanie atrybutów wierzchołków
        stride = 6 * self.sphere_vertices.itemsize
        
        # location = 0 (aPos) - pozycja X, Y, Z
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # location = 1 (aColor) - kolor R, G, B
        color_offset = 3 * self.sphere_vertices.itemsize
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(color_offset))
        glEnableVertexAttribArray(1)
        
        # Odpinanie buforów (bezpieczeństwo, nie odpinamy EBO gdy VAO jest aktywne!)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw_sphere(self, view_matrix, projection_matrix, model_matrix):
        """
        Rysuje sferę z odpowiednimi macierzami transformacji.
        """
        glUseProgram(self.shader_program)
        
        # Pobieranie lokacji zmiennych uniform w shaderze
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        model_loc = glGetUniformLocation(self.shader_program, "model")
        
        # Wgrywanie macierzy do GPU (GL_TRUE wymusza transpozycję z układu rzędowego numpy do kolumnowego w OpenGL)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view_matrix)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection_matrix)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix)
        
        # Bindowanie i rysowanie (DrawElements ponieważ użyliśmy triangulacji i indeksów EBO)
        glBindVertexArray(self.sphere_vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
