import OpenGL.GL as gl
import numpy as np
import os
from shader import Shader

class Skybox:
    """Renders a procedural 3D Skybox using a numpy-generated cubemap."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vert_path = os.path.join(base_dir, "shaders", "skybox.vert")
        frag_path = os.path.join(base_dir, "shaders", "skybox.frag")
        self.shader = Shader(vert_path, frag_path)
        
        # 1x1x1 cube centered at origin
        skybox_vertices = np.array([
            # positions          
            -1.0,  1.0, -1.0,
            -1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,

             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,

            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0
        ], dtype=np.float32)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, gl.GL_STATIC_DRAW)
        
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * skybox_vertices.itemsize, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        
        # Create starfield texture
        self.texture_id = self._generate_starfield_cubemap()
        
        # Bind texture to uniform
        self.shader.use()
        gl.glUniform1i(gl.glGetUniformLocation(self.shader.program_id, "skybox"), 0)

    def _generate_starfield_cubemap(self) -> int:
        """Generates a procedural starfield for the cubemap directly via Numpy without images."""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture_id)
        
        # For each of the 6 faces of the cube map
        for i in range(6):
            # 512x512 RGB black image
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Scatter random stars
            num_stars = 300
            x = np.random.randint(0, 512, num_stars)
            y = np.random.randint(0, 512, num_stars)
            
            # Star colors (white, bluish, yellowish)
            colors = np.random.randint(150, 256, (num_stars, 3)).astype(np.uint8)
            img[x, y] = colors
            
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.GL_RGB, 512, 512, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)
            
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        
        return texture_id

    def draw(self, view_matrix: np.ndarray, projection_matrix: np.ndarray) -> None:
        """Renders the skybox as the background."""
        # 1. Remove translation from the view matrix (last column in numpy layout)
        # This makes the skybox follow the camera perfectly so we never reach the edge
        view = view_matrix.copy()
        view[0, 3] = 0.0
        view[1, 3] = 0.0
        view[2, 3] = 0.0
        
        # 2. Change depth function so it passes when depth equals 1.0 (the max depth)
        gl.glDepthFunc(gl.GL_LEQUAL)
        
        self.shader.use()
        self.shader.set_mat4("view", view.T.copy())
        self.shader.set_mat4("projection", projection_matrix.T.copy())
        
        gl.glBindVertexArray(self.vao)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texture_id)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        gl.glBindVertexArray(0)
        
        # 3. Restore default depth function
        gl.glDepthFunc(gl.GL_LESS)
