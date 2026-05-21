# camera.py
import numpy as np
import math

class Camera:
    def __init__(self, position=np.array([0.0, 0.0, 3.0], dtype=np.float32), 
                 up=np.array([0.0, 1.0, 0.0], dtype=np.float32), 
                 yaw=-90.0, pitch=0.0):
        # Camera Attributes
        self.Position = position
        self.Front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.Up = up
        self.Right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.WorldUp = up
        
        # Euler Angles
        self.Yaw = yaw
        self.Pitch = pitch
        
        # Camera options
        self.MovementSpeed = 10.0
        self.MouseSensitivity = 0.1
        self.Zoom = 45.0

        self.update_camera_vectors()

    def get_view_matrix(self):
        """Zwraca macierz widoku używając wbudowanej funkcji lookAt zaimplementowanej w NumPy"""
        return self._look_at(self.Position, self.Position + self.Front, self.Up)
        
    def _look_at(self, eye, center, up):
        """Standardowa macierz LookAt (Row-Major)."""
        f = (center - eye)
        f_norm = np.linalg.norm(f)
        if f_norm == 0:
            return np.identity(4, dtype=np.float32)
        f = f / f_norm
        
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm == 0:
            return np.identity(4, dtype=np.float32)
        s = s / s_norm
        
        u = np.cross(s, f)
        
        m = np.identity(4, dtype=np.float32)
        # Rząd 0: Wektor Right (s) i translacja X
        m[0, 0] = s[0]; m[0, 1] = s[1]; m[0, 2] = s[2]; m[0, 3] = -np.dot(s, eye)
        # Rząd 1: Wektor Up (u) i translacja Y
        m[1, 0] = u[0]; m[1, 1] = u[1]; m[1, 2] = u[2]; m[1, 3] = -np.dot(u, eye)
        # Rząd 2: Wektor Forward (odwrócony) i translacja Z
        m[2, 0] = -f[0]; m[2, 1] = -f[1]; m[2, 2] = -f[2]; m[2, 3] = np.dot(f, eye)
        return m

    def process_keyboard(self, direction, delta_time):
        """Obsługa ruchu WASD."""
        velocity = self.MovementSpeed * delta_time
        if direction == "FORWARD":
            self.Position += self.Front * velocity
        if direction == "BACKWARD":
            self.Position -= self.Front * velocity
        if direction == "LEFT":
            self.Position -= self.Right * velocity
        if direction == "RIGHT":
            self.Position += self.Right * velocity

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Obsługa ruchu myszą."""
        xoffset *= self.MouseSensitivity
        yoffset *= self.MouseSensitivity

        self.Yaw += xoffset
        self.Pitch += yoffset

        # Zapobieganie "przekręceniu" kamery
        if constrain_pitch:
            if self.Pitch > 89.0:
                self.Pitch = 89.0
            if self.Pitch < -89.0:
                self.Pitch = -89.0

        self.update_camera_vectors()

    def update_camera_vectors(self):
        """Aktualizacja wektorów kierunkowych kamery na podstawie kątów Eulera."""
        front = np.zeros(3, dtype=np.float32)
        front[0] = math.cos(math.radians(self.Yaw)) * math.cos(math.radians(self.Pitch))
        front[1] = math.sin(math.radians(self.Pitch))
        front[2] = math.sin(math.radians(self.Yaw)) * math.cos(math.radians(self.Pitch))
        
        front_norm = np.linalg.norm(front)
        if front_norm > 0:
            self.Front = front / front_norm
        
        self.Right = np.cross(self.Front, self.WorldUp)
        self.Right = self.Right / np.linalg.norm(self.Right)
        
        self.Up = np.cross(self.Right, self.Front)
        self.Up = self.Up / np.linalg.norm(self.Up)

    def get_projection_matrix(self, width, height, near=0.1, far=100.0):
        """Oblicza i zwraca standardową macierz perspektywy (Row-Major)."""
        aspect = width / height if height > 0 else 1.0
        fovy = math.radians(self.Zoom)
        
        f = 1.0 / math.tan(fovy / 2.0)
        
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        
        return m
