import numpy as np

class PhysicsEngine:
    """Handles highly optimized N-body physics using Numpy vectorization."""

    def __init__(self, n_particles: int, center_mass: float = 10000.0, G: float = 0.5, scenario: int = 1):
        self.scenario = scenario
        self.n_particles = n_particles
        self.center_mass = center_mass
        self.G = G
        self.time_elapsed = 0.0
        
        # We will hold positions and velocities as (N, 3) arrays
        self.positions = np.zeros((n_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 3), dtype=np.float32)
        
        self._initialize_disk()

    def _initialize_disk(self) -> None:
        self.time_elapsed = 0.0
        
        if self.scenario == 1:
            # 1. Standard Single Star Orbit
            r = np.random.uniform(5.0, 50.0, self.n_particles)
            theta = np.random.uniform(0, 2 * np.pi, self.n_particles)
            self.positions[:, 0] = r * np.cos(theta)
            self.positions[:, 1] = np.random.uniform(-1.0, 1.0, self.n_particles)
            self.positions[:, 2] = r * np.sin(theta)
            
            v_mag = np.sqrt((self.G * self.center_mass) / r)
            tangent_x = -self.positions[:, 2]
            tangent_z = self.positions[:, 0]
            
            tangent_norm = np.sqrt(tangent_x**2 + tangent_z**2)
            tangent_x /= tangent_norm
            tangent_z /= tangent_norm
            
            self.velocities[:, 0] = tangent_x * v_mag
            self.velocities[:, 1] = 0.0
            self.velocities[:, 2] = tangent_z * v_mag
            
        elif self.scenario == 2:
            # 2. Binary Star System (Particles orbit slightly wider)
            r = np.random.uniform(15.0, 65.0, self.n_particles)
            theta = np.random.uniform(0, 2 * np.pi, self.n_particles)
            self.positions[:, 0] = r * np.cos(theta)
            self.positions[:, 1] = np.random.uniform(-2.0, 2.0, self.n_particles)
            self.positions[:, 2] = r * np.sin(theta)
            
            v_mag = np.sqrt((self.G * self.center_mass) / r)
            tangent_x = -self.positions[:, 2]
            tangent_z = self.positions[:, 0]
            
            tangent_norm = np.sqrt(tangent_x**2 + tangent_z**2)
            tangent_x /= tangent_norm
            tangent_z /= tangent_norm
            
            self.velocities[:, 0] = tangent_x * v_mag
            self.velocities[:, 1] = 0.0
            self.velocities[:, 2] = tangent_z * v_mag
            
        elif self.scenario == 3:
            # 3. Supernova (Dense central core bursting outwards)
            r = np.random.uniform(0.1, 3.0, self.n_particles)
            theta = np.random.uniform(0, 2 * np.pi, self.n_particles)
            phi = np.random.uniform(0, np.pi, self.n_particles)
            
            self.positions[:, 0] = r * np.sin(phi) * np.cos(theta)
            self.positions[:, 1] = r * np.sin(phi) * np.sin(theta)
            self.positions[:, 2] = r * np.cos(phi)
            
            # Massive outward velocity in all directions
            speed = np.random.uniform(15.0, 45.0, self.n_particles)
            dir_norm = self.positions / r[:, np.newaxis]
            self.velocities = dir_norm * speed[:, np.newaxis]

    def update(self, delta_time: float) -> np.ndarray:
        self.time_elapsed += delta_time
        
        if self.scenario == 1:
            # O(N) Single Star
            dir_to_center = -self.positions
            r_sq = np.sum(self.positions**2, axis=1, keepdims=True)
            r_sq_softened = r_sq + 2.0
            r = np.sqrt(r_sq_softened)
            
            accel_mag = (self.G * self.center_mass) / (r_sq_softened * r)
            acceleration = dir_to_center * accel_mag
            
            self.velocities += acceleration * delta_time
            self.positions += self.velocities * delta_time
            
        elif self.scenario == 2:
            # O(N) Binary Star System (Two points rotating at radius 5.0)
            star1_pos = np.array([np.cos(self.time_elapsed * 1.5)*5.0, 0.0, np.sin(self.time_elapsed * 1.5)*5.0])
            star2_pos = np.array([-np.cos(self.time_elapsed * 1.5)*5.0, 0.0, -np.sin(self.time_elapsed * 1.5)*5.0])
            
            # Gravity from Star 1
            dir1 = star1_pos - self.positions
            r1_sq = np.sum(dir1**2, axis=1, keepdims=True) + 2.0
            accel1 = dir1 * ((self.G * (self.center_mass / 2.0)) / (r1_sq * np.sqrt(r1_sq)))
            
            # Gravity from Star 2
            dir2 = star2_pos - self.positions
            r2_sq = np.sum(dir2**2, axis=1, keepdims=True) + 2.0
            accel2 = dir2 * ((self.G * (self.center_mass / 2.0)) / (r2_sq * np.sqrt(r2_sq)))
            
            acceleration = accel1 + accel2
            self.velocities += acceleration * delta_time
            self.positions += self.velocities * delta_time
            
        elif self.scenario == 3:
            # O(N) Supernova Remnant (Weak central gravity pulling them back slightly)
            dir_to_center = -self.positions
            r_sq = np.sum(self.positions**2, axis=1, keepdims=True)
            r_sq_softened = r_sq + 2.0
            r = np.sqrt(r_sq_softened)
            
            # Much weaker gravity (10% of original mass) representing remnant black hole
            accel_mag = (self.G * (self.center_mass * 0.1)) / (r_sq_softened * r)
            acceleration = dir_to_center * accel_mag
            
            # Drag/Friction from expanding gas cloud
            damping = -self.velocities * 0.1
            
            self.velocities += (acceleration + damping) * delta_time
            self.positions += self.velocities * delta_time
            
        return self.positions
