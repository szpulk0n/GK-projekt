# physics.py
import numpy as np

class GravityEngine:
    def __init__(self, G=1.0, softening=0.1):
        self.G = G
        self.softening = softening
        
        # Makro-obiekty (Planety)
        # N x 3 (x, y, z)
        self.planets_pos = np.empty((0, 3), dtype=np.float32)
        self.planets_vel = np.empty((0, 3), dtype=np.float32)
        self.planets_mass = np.empty((0,), dtype=np.float32)
        
        # Mikro-obiekty (Pył kosmiczny)
        # M x 3
        self.dust_pos = np.empty((0, 3), dtype=np.float32)
        self.dust_vel = np.empty((0, 3), dtype=np.float32)
        
    def add_planet(self, position, velocity, mass):
        self.planets_pos = np.vstack([self.planets_pos, np.array(position, dtype=np.float32)])
        self.planets_vel = np.vstack([self.planets_vel, np.array(velocity, dtype=np.float32)])
        self.planets_mass = np.append(self.planets_mass, np.float32(mass))
        
    def add_dust(self, positions, velocities):
        """Dodaje wsadowo (batch) cząsteczki pyłu"""
        if len(positions) == 0:
            return
        self.dust_pos = np.vstack([self.dust_pos, np.array(positions, dtype=np.float32)])
        self.dust_vel = np.vstack([self.dust_vel, np.array(velocities, dtype=np.float32)])
        
    def step(self, dt):
        num_planets = len(self.planets_pos)
        
        if num_planets > 0:
            # 1. Grawitacja Planet (N-Body)
            # Obliczenie różnic między pozycjami: r_ij = pos_j - pos_i
            # Kształt: (N, N, 3)
            pos_diff = self.planets_pos[np.newaxis, :, :] - self.planets_pos[:, np.newaxis, :]
            
            # Odległość do kwadratu + softening factor aby uniknąć dzielenia przez zero przy kolizjach
            # Kształt: (N, N)
            dist_sq = np.sum(pos_diff**2, axis=-1) + self.softening**2
            
            # 1 / r^3
            inv_dist_cube = dist_sq**(-1.5)
            # Na przekątnej mamy oddziaływania planety z samą sobą, wyzerujemy to, 
            # ale z softeningiem i tak by zniknęło mnożąc przez pos_diff=0, jednak dla bezpieczeństwa zostawiamy jak jest, pos_diff jest 0 na diagonali.
            
            # F = G * m1 * m2 * r_vec / r^3
            # Przyspieszenie a_i = F_i / m_i = Sum_j( G * m_j * r_ij / r_ij^3 )
            # Kształt: (N, N, 1) * (N, N, 3) = (N, N, 3)
            # Sumujemy po osi j (axis=1) -> wynik (N, 3)
            acc_matrix = self.G * self.planets_mass[np.newaxis, :, np.newaxis] * inv_dist_cube[:, :, np.newaxis] * pos_diff
            planets_acc = np.sum(acc_matrix, axis=1)
            
            # 2. Grawitacja na Pył (O(M * N))
            num_dust = len(self.dust_pos)
            if num_dust > 0:
                # Obliczenie różnic dla pyłu: r_ij = pos_planet_j - pos_dust_i
                # Kształt: (M, N, 3)
                dust_pos_diff = self.planets_pos[np.newaxis, :, :] - self.dust_pos[:, np.newaxis, :]
                
                dust_dist_sq = np.sum(dust_pos_diff**2, axis=-1) + self.softening**2
                dust_inv_dist_cube = dust_dist_sq**(-1.5)
                
                # a_dust_i = Sum_j( G * m_planet_j * r_ij / r_ij^3 )
                dust_acc_matrix = self.G * self.planets_mass[np.newaxis, :, np.newaxis] * dust_inv_dist_cube[:, :, np.newaxis] * dust_pos_diff
                dust_acc = np.sum(dust_acc_matrix, axis=1)
                
                # Aktualizacja prędkości i pozycji pyłu (Metoda Eulera/Semi-implicit Euler)
                self.dust_vel += dust_acc * dt
                self.dust_pos += self.dust_vel * dt
            
            # Aktualizacja prędkości i pozycji planet
            self.planets_vel += planets_acc * dt
            self.planets_pos += self.planets_vel * dt
