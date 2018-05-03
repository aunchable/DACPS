import numpy as np
import random
from collections import namedtuple


class ColloidalSystem:
    def __init__(self, world_dims, type_infos, type_counts, lj_corr_matrix):
        self.world_dims = world_dims
        self.num_particles = np.sum(type_counts)
        self.particle_types = np.array([k for k, type_count in enumerate(type_counts) for j in range(type_count) ])
        self.lj_corr_matrix = lj_corr_matrix
        self.temp = 0
        self.light_frac = 1.0
        self.state = np.zeros(shape=(self.num_particles, 6))

    def step(self, dt, light_mask=[]):
        new_state = np.zeros(shape=(self.num_particles, 6))

        # LJ forces
        for p_idx in range(self.num_particles):
            #TODO: compute pairwise potentials, update position

        # Brownian motion

        # Light illumination

        self.state = new_state
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
