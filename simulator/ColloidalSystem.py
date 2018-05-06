import numpy as np
import random
from collections import namedtuple

kB = 0.00138064852 # in nm
VISCOSITY = 1.0
LIGHT_ACCEL_TIME = 8.0
PARTICLE_DENSITY = 1.228 * 10^-15 # 1.228 g/cm3 converted to kg/nm^3

class ColloidalSystem:
    def __init__(self, world_dims, type_infos, type_counts, lj_corr_matrix, target_assembly):
        # World definition
        self.world_dims = world_dims
        self.num_particles = np.sum(type_counts)
        self.particle_types = np.array([k for k, type_count in enumerate(type_counts) for j in range(type_count)])

        # Type parameters [Global System Parameters]
        self.num_types = len(type_infos)
        self.lj_corr_matrix = lj_corr_matrix
        self.type_radii = [t[1] for t in type_infos]
        self.type_light_propensities = [t[2] for t in type_infos]
        self.type_max_vel = self.get_type_max_vel()
        self.type_mass = self.get_type_mass()

        # Simulation parameters [Assembly System Parameters]
        self.temp = 0
        self.light_frac = 1.0

        # Simulation execution information
        self.state = np.zeros(shape=(self.num_particles, 6))
        self.time = 0.0

        # For learning
        self.target_assembly = target_assembly


    def step(self, dt, light_mask=[]):
        new_state = np.zeros(shape=(self.num_particles, 6))

        accel = np.zeros(shape=(self.num_particles, 3))

        # LJ forces
        for p1_idx in range(self.num_particles):
            for p2_idx in range(p1_idx + 1, self.num_particles):
                A, B = self.lj_corr_matrix[self.particle_types[p1_idx]][self.particle_types[p2_idx]]
                dr = self.state[p1_idx, :2] - self.state[p2_idx, :2]
                dr2 = np.dot(dr, dr)
                acc = -(A * np.power((1/dr2), 6)- B * np.power((1/dr2), 3)) * dr / dr2
                accel[p1_idx][:2] += acc
                accel[p2_idx][:2] -= acc

        # Brownian motion
        for p_idx in range(self.num_particles):
            r = self.type_radii[self.particle_types[p_idx]]
            D = kB * self.temp / (6 * np.pi * VISCOSITY * r)
            Dr = kB * self.temp / (8 * np.pi * VISCOSITY * np.power(r, 3))
            dx, dy = np.random.normal(0, np.sqrt(2 * D * dt), 2)
            dphi = np.random.normal(0, np.sqrt(2 * Dr * dt), 1)
            # a = (2/t^2) * (d - v_i * t)
            accel[p_idx] += (2 / (dt * dt)) * np.array([d - self.state[p_idx, i + 3] * dt for i, d in enumerate([dx, dy, dphi])]

        # Light illumination
        for p_idx in range(self.num_particles):
            if light_mask[p_idx]:
                type = self.particle_types[p_idx]
                o = self.state[p_idx][2]
                mult_factor = (self.light_frac) / (self.light_frac + self.type_light_propensities[type])
                final_velocity = mult_factor  * self.type_max_vel[type] * np.array([np.cos(o), np.sin(o)])
                accel[p_idx][:2] += (final_velocity - self.state[p_idx, 3:5]) / LIGHT_ACCEL_TIME

        # Drag force [Stoke's Law]
        for p_idx in range(self.num_particles):
            r = self.type_radii[self.particle_types[p_idx]]
            D = kB * self.temp / (6 * np.pi * VISCOSITY * r)
            Dr = kB * self.temp / (8 * np.pi * VISCOSITY * np.power(r, 3))
            mass = self.type_mass[self.particle_types[p_idx]]
            moment_of_inertia = 0.4 * mass * r * r
            accel[p_idx][0] -= D * self.state[p_idx][3] / mass
            accel[p_idx][1] -= D * self.state[p_idx][4] / mass
            accel[p_idx][2] -= Dr * self.state[p_idx][5] * r / moment_of_inertia

        # Update positions/velocities
        for p_idx in range(self.num_particles):
            new_state[p_idx, 3:] = self.state[p_idx, 3:] + accel[p_index] * dt
            new_state[p_idx, :3] = self.state[p_idx, :3] + new_state[p_index, 3:] * dt
            while new_state[p_idx][2] < 0.0:
                new_state[p_idx][2] += 2 * np.pi
            while new_state[p_idx][2] > 2 * np.pi:
                new_state[p_idx][2] -= 2 * np.pi

        # Elastic world boundaries
        for p_idx in range(self.num_particles):
            if new_state[p_idx][0] < 0:
                new_state[p_idx][0] = -new_state[p_idx][0]
            elif new_state[p_idx][0] > self.world_dims[0]:
                new_state[p_idx][0] = 2 * self.world_dims[0] - new_state[p_idx][0]

            if new_state[p_idx][1] < 0:
                new_state[p_idx][1] = -new_state[p_idx][1]
            elif new_state[p_idx][1] > self.world_dims[1]:
                new_state[p_idx][1] = 2 * self.world_dims[1] - new_state[p_idx][1]

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

    def set_target_assembly(self, target_assembly):
        self.target_assembly = target_assembly


    def get_type_max_vel(self):
        type_max_accel = []
        for i in range(len(self.num_types)):
            type_max_vel.append(150000.0)
        return type_max_vel

    def get_type_mass(self):
        type_mass = []
        for i in range(len(self.num_types)):
            type_mass.append(PARTICLE_DENSITY * 4 * np.pi * self.type_radii[i]^3 / 3.0)
        return type_mass


    def get_reward(self):
        # TODO(anish)
