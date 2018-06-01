import numpy as np
import scipy.linalg as la
import random
from collections import namedtuple

kB = 1.38064852e-5 # 1.38 * 10^-23 m^2 kg / (s^2 K) = 1.38 * 10^-5 nm^2 kg / (s^2 K)
VISCOSITY = 8.9e-13 # 8.90 * 10^-4 Pa-s = 8.90 * 10-13 kg/(nm s^2)
LIGHT_ACCEL_TIME = 1.0 # in s
PARTICLE_DENSITY = 1.228e-24 # 1.228 g/cm3 = 1.228 * 10^-24 kg/nm^3
WATER_DENSITY = 997e-27 # 997 kg/m3 = 997 * 10^-27 kg/nm3

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

        # print(self.type_max_vel)
        # print(self.type_mass)

        # Simulation parameters [Assembly System Parameters]
        self.temp = 0
        self.light_frac = 1.0

        # Simulation execution information
        self.state = np.zeros(shape=(self.num_particles, 6))
        self.time = 0.0

        # For learning
        self.target_assembly = target_assembly

        self.Rx, self.Ry = np.average(target_assembly, axis=0)
        self.Rtheta = np.array([np.arctan2(x - self.Rx, y - self.Ry) for (x, y) in target_assembly])
        self.Rindex = np.argsort(self.Rtheta)
        self.Rdelta = np.array([(x - self.Rx)*(x - self.Rx) + (y - self.Ry)*(y - self.Ry) for (x, y) in target_assembly])
        self.Rstart = np.where(self.Rindex == np.argmax(self.Rdelta))[0][0]


    def set_state(self, init_state):
        self.state = init_state


    def step(self, dt, light_mask=[]):
        # print('step')
        # print(self.state[0][3:])
        new_state = np.zeros(shape=(self.num_particles, 6))

        accel = np.zeros(shape=(self.num_particles, 3))

        # print(light_mask)
        # print(accel[0])
        # LJ forces
        # For reference, (eps, sig) for argon-argon is (0.997 kJ/mol, 4.0 angstroms)
        for p1_idx in range(self.num_particles):
            for p2_idx in range(p1_idx + 1, self.num_particles):
                eps, sig = self.lj_corr_matrix[self.particle_types[p1_idx]][self.particle_types[p2_idx]]
                dxyz = self.state[p1_idx, :2] - self.state[p2_idx, :2]
                dr = np.sqrt(np.dot(dxyz, dxyz))
                minusdUrdr = - 48 * eps * (np.power(sig / dr, 12) - np.power(sig / dr, 6)) / dr
                force = minusdUrdr * dxyz
                accel[p1_idx][:2] += (force / self.type_mass[self.particle_types[p1_idx]])
                accel[p2_idx][:2] -= (force / self.type_mass[self.particle_types[p2_idx]])
        # print(accel[0])

        # Brownian motion
        # Linear = Gaussian with variance 2*D*t
        #     D = kB * T / (6 * pi * eta * radius)
        # Angular = Gaussian with variance 2*Dr*t
        #     D = kB * T / (8 * pi * eta * radius^3)
        # m*a = sqrt(2 * gamma * kB * temp) R(t), with R(t) = N(0, dt)
        for p_idx in range(self.num_particles):
            r = self.type_radii[self.particle_types[p_idx]]
            gamma = 6 * np.pi * VISCOSITY * r
            gammar = 8 * np.pi * VISCOSITY * np.power(r, 3)
            D = kB * self.temp / gamma
            Dr = kB * self.temp / gammar

            mass = self.type_mass[self.particle_types[p_idx]]
            moment_of_inertia = 0.4 * mass * r * r

            ax = gamma * np.sqrt(2 * D) * np.random.normal(0, dt**2) / mass
            ay = gamma * np.sqrt(2 * D) * np.random.normal(0, dt**2) / mass
            aphi = gammar * np.sqrt(2 * Dr) * np.random.normal(0, dt**2) / moment_of_inertia
            accel[p_idx] += np.array([ax, ay, aphi])

        # print(accel[0])
        # Light illumination
        for p_idx in range(self.num_particles):
            if light_mask[p_idx]:
                particle_type = self.particle_types[p_idx]
                o = self.state[p_idx][2]
                mult_factor = (self.light_frac) / (self.light_frac + self.type_light_propensities[particle_type])
                final_velocity = mult_factor * self.type_max_vel[particle_type] * np.array([np.cos(o), np.sin(o)])
                accel[p_idx][:2] += (final_velocity - self.state[p_idx, 3:5]) / LIGHT_ACCEL_TIME

        # print(accel[0])
        # a = np.copy(accel[0])
        # Drag force [Stoke's Law]
        for p_idx in range(self.num_particles):
            r = self.type_radii[self.particle_types[p_idx]]
            mass = self.type_mass[self.particle_types[p_idx]]
            moment_of_inertia = 0.4 * mass * r * r

            # C = 6 * np.pi * VISCOSITY * r
            # Cr = 8 * np.pi * VISCOSITY * np.power(r, 3)
            # accel[p_idx][0] -= C * self.state[p_idx][3] / mass
            # accel[p_idx][1] -= C * self.state[p_idx][4] / mass
            # accel[p_idx][2] -= Cr * self.state[p_idx][5] * r / moment_of_inertia

            D = 0.47 * WATER_DENSITY * np.pi * r * r * 0.5
            Dr = 0.047 * WATER_DENSITY * np.pi * r * r * 0.5
            mass = self.type_mass[self.particle_types[p_idx]]
            # print(D, D * self.state[p_idx][3]**2 / mass)
            moment_of_inertia = 0.4 * mass * r * r
            accel[p_idx][0] -= np.sign(self.state[p_idx][3]) * D * self.state[p_idx][3]**2 / mass
            accel[p_idx][1] -= np.sign(self.state[p_idx][4]) * D * self.state[p_idx][4]**2 / mass
            accel[p_idx][2] -= np.sign(self.state[p_idx][5]) * Dr * self.state[p_idx][5]**2 * np.power(r, 3) / moment_of_inertia
        # print(self.state[0][3:])
        # print([c - d for c,d, in zip(accel[0], a)])
        # print(accel[0])

        # assert(False)
        # Update positions/velocities
        for p_idx in range(self.num_particles):
            new_state[p_idx, 3:] = self.state[p_idx, 3:] + accel[p_idx] * dt
            new_state[p_idx, :3] = self.state[p_idx, :3] + new_state[p_idx, 3:] * dt
            while new_state[p_idx][2] < 0.0:
                new_state[p_idx][2] += 2 * np.pi
            while new_state[p_idx][2] > 2 * np.pi:
                new_state[p_idx][2] -= 2 * np.pi

        # Elastic world boundaries
        for p_idx in range(self.num_particles):
            if new_state[p_idx][0] < 0:
                new_state[p_idx][0] = -new_state[p_idx][0]
                new_state[p_idx][3] = -new_state[p_idx][3]
            elif new_state[p_idx][0] > self.world_dims[0]:
                new_state[p_idx][0] = 2 * self.world_dims[0] - new_state[p_idx][0]
                new_state[p_idx][3] = -new_state[p_idx][3]

            if new_state[p_idx][1] < 0:
                new_state[p_idx][1] = -new_state[p_idx][1]
                new_state[p_idx][4] = -new_state[p_idx][4]
            elif new_state[p_idx][1] > self.world_dims[1]:
                new_state[p_idx][1] = 2 * self.world_dims[1] - new_state[p_idx][1]
                new_state[p_idx][4] = -new_state[p_idx][4]

        # print(self.state[0][2], self.state[0][3:6], accel[0])
        # print(new_state[0])
        self.state = new_state
        self.time += dt
        return


    def get_state(self):
        return self.state


    def random_initialization(self):
        self.state[:, 0] = np.random.random((self.num_particles, )) * self.world_dims[0]
        self.state[:, 1] = np.random.random((self.num_particles, )) * self.world_dims[1]
        self.state[:, 2] = np.random.random((self.num_particles, )) * 2 * np.pi
        self.state[:, 3:] = 0
        return


    def set_temperature(self, temp):
        self.temp = temp

    def set_light_frac(self, light_frac):
        self.light_frac = light_frac

    def set_target_assembly(self, target_assembly):
        self.target_assembly = target_assembly


    def get_type_max_vel(self):
        type_max_vel = []
        for i in range(self.num_types):
            type_max_vel.append(15000.0)
        return type_max_vel

    def get_type_mass(self):
        type_mass = []
        for i in range(self.num_types):
            type_mass.append(PARTICLE_DENSITY * 4 * np.pi * self.type_radii[i]**3 / 3.0)
        return type_mass

    def get_reward(self):
        # TODO(anish) - improve this if needed
        # shape = self.state[:,:2]
        # xbar, ybar = np.average(shape, axis=0)
        # Stheta = np.array([np.arctan2(x - xbar, y - ybar) for (x, y) in shape])
        # Sindex = np.argsort(Stheta)
        # Sdelta = np.array([(x - xbar)*(x - xbar) + (y - ybar)*(y - ybar) for (x, y) in shape])
        # Sstart = np.where(Sindex == np.argmax(Sdelta))[0][0]
        #
        # deltaLoss = np.sum([(self.Rdelta[self.Rindex[(self.Rstart + i) % self.num_particles]] - Sdelta[Sindex[(Sstart + i) % self.num_particles]])**2 for i in range(self.num_particles)])
        # thetaLoss = np.sum([(self.Rtheta[self.Rindex[(self.Rstart + i) % self.num_particles]] - Stheta[Sindex[(Sstart + i) % self.num_particles]])**2 for i in range(self.num_particles)])
        #
        # return 0 - deltaLoss - thetaLoss

        positions = self.state[:, :2]
        target_positions = self.target_assembly
        #
        # centered_positions = positions - np.mean(positions, axis = 0)
        # centered_target_positions = target_positions - np.mean(target_positions, axis = 0)
        #
        # transform = np.array(la.orthogonal_procrustes(centered_positions, centered_target_positions)[0])
        # distance = np.linalg.norm(transform - np.identity(2))
        distance = -np.abs(sum(positions[0] - target_positions[0] ))
        return distance
