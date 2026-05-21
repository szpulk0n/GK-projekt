import OpenGL.GL as gl
from typing import Optional

class Shader:
    """Compiles and links vertex and fragment shaders."""

    def __init__(self, vertex_path: str, fragment_path: str):
        self.program_id = self._compile_and_link(vertex_path, fragment_path)

    def _read_source(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    def _compile_shader(self, source: str, shader_type: int) -> int:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        
        success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not success:
            log = gl.glGetShaderInfoLog(shader)
            raise RuntimeError(f"Shader compilation failed: {log.decode('utf-8')}")
            
        return shader

    def _compile_and_link(self, vertex_path: str, fragment_path: str) -> int:
        vertex_src = self._read_source(vertex_path)
        fragment_src = self._read_source(fragment_path)
        
        vertex_shader = self._compile_shader(vertex_src, gl.GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_src, gl.GL_FRAGMENT_SHADER)
        
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        
        success = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
        if not success:
            log = gl.glGetProgramInfoLog(program)
            raise RuntimeError(f"Program linking failed: {log.decode('utf-8')}")
            
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        return program

    def use(self) -> None:
        gl.glUseProgram(self.program_id)

    def set_mat4(self, name: str, matrix) -> None:
        location = gl.glGetUniformLocation(self.program_id, name)
        gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, matrix)
