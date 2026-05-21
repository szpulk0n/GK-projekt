import numpy as np

class PhysicsEngine:
    """Handles highly optimized N-body physics using Numpy vectorization."""

    def __init__(self, n_particles: int, center_mass: float = 10000.0, G: float = 0.5):
        self.n_particles = n_particles
        self.center_mass = center_mass
        self.G = G
        
        # We will hold positions and velocities as (N, 3) arrays
        self.positions = np.zeros((n_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 3), dtype=np.float32)
        
        self._initialize_disk()

    def _initialize_disk(self) -> None:
        """Spawns particles in a wide disk around the central star with orbital velocities."""
        # Random radius from center (e.g., between 5 and 50)
        r = np.random.uniform(5.0, 50.0, self.n_particles)
        # Random angle around the Y axis
        theta = np.random.uniform(0, 2 * np.pi, self.n_particles)
        
        # Convert polar to Cartesian for positions (XZ plane mostly, slight Y variation for thickness)
        self.positions[:, 0] = r * np.cos(theta)
        self.positions[:, 1] = np.random.uniform(-1.0, 1.0, self.n_particles) # Slight disk thickness
        self.positions[:, 2] = r * np.sin(theta)
        
        # Calculate orbital velocity for circular orbit: v = sqrt(G * M / r)
        v_mag = np.sqrt((self.G * self.center_mass) / r)
        
        # Tangent vector is perpendicular to position vector in the XZ plane.
        # If pos is (x, y, z), tangent is (-z, 0, x) normalized.
        tangent_x = -self.positions[:, 2]
        tangent_z = self.positions[:, 0]
        
        # Normalize tangents
        tangent_norm = np.sqrt(tangent_x**2 + tangent_z**2)
        tangent_x /= tangent_norm
        tangent_z /= tangent_norm
        
        # Apply velocities
        self.velocities[:, 0] = tangent_x * v_mag
        self.velocities[:, 1] = 0.0 # mostly flat orbit
        self.velocities[:, 2] = tangent_z * v_mag

    def update(self, delta_time: float) -> np.ndarray:
        """
        Updates positions and velocities for one timestep.
        Returns the new position array to be sent to the GPU.
        """
        # --- O(N) Gravity Calculation ---
        # Gravity only from the center at (0,0,0)
        
        # Vector pointing from particle to center
        # Since center is at (0,0,0), this is just -position
        dir_to_center = -self.positions
        
        # Distance squared (dot product of position with itself along axis 1)
        r_sq = np.sum(self.positions**2, axis=1, keepdims=True)
        
        # Prevent division by zero or explosive forces near center
        # Add a small softening parameter (epsilon)
        epsilon = 2.0
        r_sq_softened = r_sq + epsilon
        
        # Distance r
        r = np.sqrt(r_sq_softened)
        
        # Force magnitude per unit mass: F = G * M / r^2
        # Acceleration a = F / m = G * M / r^2
        # Direction is dir_to_center / r
        # Therefore, vector a = dir_to_center * (G * M / r^3)
        accel_mag = (self.G * self.center_mass) / (r_sq_softened * r)
        
        acceleration = dir_to_center * accel_mag
        
        # Semi-implicit Euler integration
        self.velocities += acceleration * delta_time
        self.positions += self.velocities * delta_time
        
        return self.positions
