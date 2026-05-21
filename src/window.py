import glfw
import sys

class Window:
    """Manages the GLFW window and OpenGL context."""

    def __init__(self, width: int, height: int, title: str):
        if not glfw.init():
            print("Failed to initialize GLFW")
            sys.exit(1)
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            sys.exit(1)
            
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        
        self.last_frame_time = glfw.get_time()
        self.delta_time = 0.0

    def _framebuffer_size_callback(self, window, width: int, height: int) -> None:
        import OpenGL.GL as gl
        gl.glViewport(0, 0, width, height)

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def update(self) -> float:
        """Swaps buffers, polls events, and returns delta time."""
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
        current_time = glfw.get_time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        return self.delta_time

    def terminate(self) -> None:
        glfw.terminate()
