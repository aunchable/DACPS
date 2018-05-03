import numpy as np
import random
from collections import namedtuple


class ColloidalSystem:
    def __init__(self, world_dims, type_infos, type_counts, lj_corr_matrix):
        self.world_dims = world_dims
        self.num_types = len(type_infos)
        self.num_particles = np.sum(type_counts)
        self.particle_types = np.array([k for k, type_count in enumerate(type_counts) for j in range(type_count)])
        self.lj_corr_matrix = lj_corr_matrix
        self.type_radii = [t[1] for t in type_infos]
        self.type_light_propensities = [t[2] for t in type_infos]
        self.temp = 0
        self.light_frac = 1.0
        self.state = np.zeros(shape=(self.num_particles, 6))
        self.time = 0.0
        self.type_max_accel = self.set_type_max_accel()

    def step(self, dt, light_mask=[]):
        new_state = np.zeros(shape=(self.num_particles, 6))

        accel = np.zeros(shape=(self.num_particles, 3))

        # LJ forces
        for p1_idx in range(self.num_particles):
            for p2_idx in range(p1_idx + 1, self.num_particles):
                A, B = self.lj_corr_matrix[self.particle_types[p1_idx]][self.particle_types[p2_idx]]
                dr = self.state[p1_idx, :2] - self.state[p2_idx, :2]
                dr2 = np.dot(dr, dr)
                acc = -(A*(1/dr2)^6 - B*(1/dr2)^3) * dr / dr2
                accel[p1_idx][:2] += acc
                accel[p2_idx][:2] -= acc

        # TODO: Brownian motion
        for p_idx in range(self.num_particles):
            cnt = 0

        # Light illumination
        for p_idx in range(self.num_particles):
            if light_mask[p_idx]:
                type = self.particle_types[p_idx]
                o = self.state[p_idx][2]
                mult_factor = (self.light_frac) / (self.light_frac + self.type_light_propensities[type])
                accel[p_idx][:2] += mult_factor  * self.type_max_accel[type] * np.array([np.cos(o), np.sin(o)])


        # Update positions/velocities
        for p_idx in range(self.num_particles):
            new_state[p_idx, 3:] = self.state[p_idx, 3:] + accel[p_index] * dt
            new_state[p_idx, :3] = self.state[p_idx, :3] + new_state[p_index, 3:] * dt

        self.state = new_state
        self.time += dt
        return

    def get_state(self):
        return self.state

    def random_initialization(self):
        self.state[:, 0] = np.random((self.num_particles, _)) * self.world_dims[0]
        self.state[:, 1] = np.random((self.num_particles, _)) * self.world_dims[1]
        self.state[:, 2] = np.random((self.num_particles, _)) * 2 * np.pi
        self.state[:, 3:] = 0
        return

    def set_temperature(self, temp):
        self.temp = temp

    def set_light_frac(self, light_frac):
        self.light_frac = light_frac

    def set_type_max_accel():
        # TODO: calculations
        type_max_accel = []
        for i in range(len(self.num_types)):
            type_max_accel.append(0.0)
        self.set_type_max_accel = type_max_accel
