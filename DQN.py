import math
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import namedtuple
from simulator.ColloidalSystem import ColloidalSystem
from simulator.Visualizer import Visualizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle
import h5py

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, num_actions, num_particles, state_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(int(state_size*num_particles), 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x.view(x.size(0), -1))

class DQNAgent():
    def __init__(self, cs):
        self.cs = cs
        self.simple_test_flag = 0
        self.viz = Visualizer(self.cs)
        self.num_particles = cs.num_particles
        self.state_size = int(3)
        self.num_actions = int(2**(self.num_particles) )

        if self.simple_test_flag:
            self.num_actions = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 10
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.BUFFER_SIZE = 20
        self.policy_net = DQN(self.num_actions, self.num_particles, self.state_size).to(self.device)
        self.target_net = DQN(self.num_actions, self.num_particles, self.state_size).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.BUFFER_SIZE)
        self.steps_done = 0
        self.num_episodes = 10000
        self.num_time_steps = 200
        self.reward_list = []
        self.final_result_per_episode = []

    # Epsilon-greedy action selection using policy_net
    def select_action(self, state):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.num_actions - 1)]], device = self.device, dtype = torch.long)

    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # max_a' Q^ (psi_new, a' | theta-)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  # gamma max_a' Q^ (psi_new, a' | theta-) + r

        # Compute Huber loss between gamma max_a' Q^ (psi_new, a' | theta-) + r and Q(psi_old, a* | theta)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_model(self):
        f = h5py.File('./logging/logs' + str(LOGNUM) + '.txt', "w", libver='latest')
        stats_grp = f.create_group("statistics")
        dset_avgq = stats_grp.create_dataset("ep_qval", (self.num_episodes,), dtype='f')
        dset_rewards = stats_grp.create_dataset("ep_reward", (self.num_episodes,), dtype='f')
        f.swmr_mode = True # NECESSARY FOR SIMULTANEOUS READ/WRITE

        for i_episode in range(self.num_episodes):
            print("EPISODE: " + str(i_episode))
            # Initialize the environment and state
            self.cs.random_initialization()


            state = self.cs.get_state()[:, :3]
            state = [item for sublist in state for item in sublist]
            state = torch.tensor([state], device=self.device, dtype = torch.float)
            r_init = self.cs.get_reward()
            r_old = r_init

            for t in range(self.num_time_steps):

                # Select and perform an action
                action = self.select_action(state)

                self.steps_done += 1
                light_mask = (( (((action.item() & (1 << np.arange(self.num_particles)))) > 0).astype(int) ).tolist() )

                # Add visualization
                if t % 10 == 0:
                    self.viz.update()

                if self.simple_test_flag:
                    positions = self.cs.state[:, :2] # temporary

                    if action.item() == 0:
                        positions[0][0]+=8
                    elif action.item() == 1:
                        positions[0][0]-=8
                    elif action.item() == 2:
                        positions[0][1]+=8
                    elif action.item() == 3:
                        positions[0][1]-=8
                    else:
                        print("ERROR")
                else:
                    for j in range(200):
                        self.cs.step(0.001, light_mask)

                # Add visualization
                if t % 10 == 0:
                    time.sleep(0.1)

                # Get reward
                r_new = self.cs.get_reward()
                reward = torch.tensor([r_new - r_old], device=self.device, dtype = torch.float)
                self.reward_list.append(r_new - r_old)

                # Observe new state
                next_state = self.cs.get_state()[:, :3]
                next_state = [item for sublist in next_state for item in sublist]
                next_state = torch.tensor([next_state], device=self.device, dtype = torch.float)

                # Compute TD Error
                state_action_val = self.policy_net(state).gather(1, action)

                max_target_val = self.target_net(next_state).max(1)[0].detach()
                td_error = max_target_val*self.GAMMA + reward

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

                r_old = r_new

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            r_episode = self.cs.get_reward() - r_init
            self.final_result_per_episode.append(r_episode)
            print("Episode Reward: " + str(r_episode))

            if len(self.memory) >= self.BATCH_SIZE:
                transitions = self.memory.sample(self.BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                dset_avgq[i] = state_action_values.mean()
            else:
                dset_avgq[i] = 0.0

            dset_rewards[i] = r_episode

            dset_avgq.flush()
            dset_rewards.flush()

        pickle.dump(self.final_result_per_episode, open( "episode_rewards.p", "wb" ))

        # print('Complete')

if __name__ == "__main__":
    worldsize = [800, 800]

    type_infos = [
        # id, radius, propensity
        ["jeb", 1100, 420],
        ["shiet", 1200, 420],
        ["goteem", 1300, 420]
    ]

    # type_counts = [1]
    type_counts = [1, 1, 1]

    # lj_corr_matrix = [[(np.random.random(), np.random.random())]]

    lj_corr_matrix = [
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())]
    ]

    target_assembly = np.array([[0, 0]])

    cs = ColloidalSystem(worldsize,
                         type_infos,
                         type_counts,
                         lj_corr_matrix,
                         target_assembly)
    cs.set_temperature(300)

    agent = DQNAgent(cs)
    agent.train_model()
