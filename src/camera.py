import numpy as np
import math

class Camera:
    """Free-fly 3D camera module."""

    def __init__(self, position=(0.0, 0.0, 5.0)):
        self.position = np.array(position, dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.yaw = -90.0
        self.pitch = 0.0

        self.movement_speed = 5.0
        self.mouse_sensitivity = 0.1
        self.fov = 45.0
        
        self._update_camera_vectors()

    def get_view_matrix(self) -> np.ndarray:
        """Returns the LookAt view matrix."""
        f = self.front
        s = np.cross(f, self.world_up)
        
        # Avoid division by zero
        s_norm = np.linalg.norm(s)
        if s_norm > 0:
            s /= s_norm
            
        u = np.cross(s, f)

        mat = np.eye(4, dtype=np.float32)
        mat[0, :3] = s
        mat[1, :3] = u
        mat[2, :3] = -f
        mat[0, 3] = -np.dot(s, self.position)
        mat[1, 3] = -np.dot(u, self.position)
        mat[2, 3] = np.dot(f, self.position)
        
        return mat

    def get_projection_matrix(self, aspect_ratio: float, near=0.1, far=100.0) -> np.ndarray:
        """Returns the perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect_ratio
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2.0 * far * near) / (near - far)
        matrix[3, 2] = -1.0
        return matrix

    def process_keyboard(self, direction: str, delta_time: float) -> None:
        """Processes WASD movement."""
        velocity = self.movement_speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity

    def process_mouse_movement(self, xoffset: float, yoffset: float, constrain_pitch=True) -> None:
        """Processes pitch and yaw from mouse movement."""
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self._update_camera_vectors()

    def _update_camera_vectors(self) -> None:
        """Calculates the new front, right, and up vectors."""
        front = np.zeros(3, dtype=np.float32)
        front[0] = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front[1] = math.sin(math.radians(self.pitch))
        front[2] = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, self.world_up)
        
        right_norm = np.linalg.norm(self.right)
        if right_norm > 0:
            self.right /= right_norm
            
        self.up = np.cross(self.right, self.front)
        
        up_norm = np.linalg.norm(self.up)
        if up_norm > 0:
            self.up /= up_norm
