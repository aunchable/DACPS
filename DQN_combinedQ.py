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
from torch.autograd import Variable
import h5py

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

LOGNUMBER = 1

def one_hot_encode(i, n):
    return (( (((int(i) & (1 << np.arange(n)))) > 0).astype(int) ).tolist() )

def convert_to_int(encoded):
    encoded = encoded.numpy()[0]
    return sum(encoded*np.array([2**i for i in range(len(encoded))]) )

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

    def __init__(self, num_actions, num_particles, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(int(state_size*num_particles + action_size), 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.view(x.size(0), -1)

class DQNAgent():
    def __init__(self, cs):
        self.cs = cs
        self.simple_test_flag = 0
        self.viz = Visualizer(self.cs)
        self.num_particles = cs.num_particles
        self.state_size = int(3)
        self.action_size = cs.num_particles
        self.num_actions = int(2**(self.num_particles) )

        if self.simple_test_flag:
            self.num_actions = 5
            self.action_size = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 256
        self.STATS_BATCH_SIZE = 1024
        self.GAMMA = 0.999
        self.EPS_START = 1
        self.EPS_END = 0.1
        self.EPS_DECAY = 500
        self.TARGET_UPDATE = 10
        self.BUFFER_SIZE = 10000
        self.policy_net = DQN(self.num_actions, self.num_particles, self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.num_actions, self.num_particles, self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.BUFFER_SIZE)
        self.steps_done = 0
        self.num_episodes = 100000
        self.num_time_steps = 250
        self.reward_list = []
        self.final_result_per_episode = []

    # Epsilon-greedy action selection using policy_net
    def select_action(self, state):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

        max_Q_val = -1000000000000

        max_action_tensor = torch.tensor([  one_hot_encode(0, self.action_size) ], device = self.device, dtype = torch.float)

        for action in range(self.num_actions):
            action_tensor = torch.tensor([   one_hot_encode(action, self.action_size)    ], device = self.device, dtype = torch.float)

            action_val = self.policy_net(state, action_tensor)
            if action_val > max_Q_val:
                max_Q_val = action_val
                max_action_tensor = action_tensor

        if sample > eps_threshold:
            with torch.no_grad():
                return max_action_tensor
        else:
            return torch.tensor([   one_hot_encode(random.randint(0, self.num_actions - 1), self.action_size)   ], device = self.device, dtype = torch.float)

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
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        state_action_values = self.policy_net(state_batch, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        for state_index in range(len(non_final_next_states)):
            # if non_final_mask[state_index]:
            max_action_val_for_state = -100000000000

            for a in range(self.num_actions):
                action_tensor = torch.tensor([   one_hot_encode(a, self.action_size)     ], device = self.device, dtype = torch.float)
                action_val = self.target_net(non_final_next_states[state_index].view(-1, int(self.state_size*self.num_particles) ), action_tensor).detach() # CHECK IF THIS PART WORKS

                if action_val > max_action_val_for_state:
                    max_action_val_for_state = action_val

            next_state_values[state_index] = max_action_val_for_state

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  # gamma max_a' Q^ (psi_new, a' | theta-) + r

        # Compute Huber loss between gamma max_a' Q^ (psi_new, a' | theta-) + r and Q(psi_old, a* | theta)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10000, 10000)
        self.optimizer.step()

    def train_model(self):
        f = h5py.File('./logging/logs' + str(LOGNUMBER) + '.txt', "w", libver='latest')
        stats_grp = f.create_group("statistics")
        dset_q = stats_grp.create_dataset("ep_q", (self.num_episodes, self.STATS_BATCH_SIZE), dtype='f')
        dset_rewards = stats_grp.create_dataset("ep_reward", (self.num_episodes,), dtype='f')
        if self.simple_test_flag:
            dset_goodq = stats_grp.create_dataset("ep_goodq", (self.num_episodes,), dtype='f')
            dset_badq = stats_grp.create_dataset("ep_badq", (self.num_episodes,), dtype='f')


        f.swmr_mode = True # NECESSARY FOR SIMULTANEOUS READ/WRITE

        for i_episode in range(self.num_episodes):
            print("EPISODE: " + str(i_episode))
            # Initialize the environment and state
            self.cs.random_initialization()
            # self.cs.set_state(np.array([
            #     [400, 400, 0, 0, 0, 0],
            # ]))

            state = self.cs.get_state()[:, :3]
            state = [item for sublist in state for item in sublist]
            state = torch.tensor([state], device=self.device, dtype = torch.float)
            r_init = self.cs.get_reward()
            r_old = r_init

            for t in range(self.num_time_steps):

                # Select and perform an action
                action = self.select_action(state)
                # print("ACTION")
                # print(action)
                int_action = convert_to_int(action)

                self.steps_done += 1
                light_mask = (( (((int(int_action) & (1 << np.arange(self.num_particles)))) > 0).astype(int) ).tolist() )

                # Add visualization
                # if t % 10 == 0:
                #     self.viz.update()

                if self.simple_test_flag:
                    positions = self.cs.state[:, :2] # temporary

                    if int_action.item() == 0:
                        # print("RIGHT")
                        positions[0][0]+=8
                    elif int_action.item() == 1:
                        # print("LEFT")
                        positions[0][0]-=8
                    elif int_action.item() == 2:
                        # print("DOWN")
                        positions[0][1]+=8
                    elif int_action.item() == 3:
                        # print("UP")
                        positions[0][1]-=8
                else:
                    for j in range(200):
                        self.cs.step(0.001, light_mask)

                # Add visualization
                # if t % 10 == 0:
                #     time.sleep(0.1)

                # Get reward
                r_new = self.cs.get_reward()
                reward = torch.tensor([r_new - r_old], device=self.device, dtype = torch.float)
                self.reward_list.append(r_new - r_old)

                # Observe new state
                next_state = self.cs.get_state()[:, :3]
                next_state = [item for sublist in next_state for item in sublist]
                next_state = torch.tensor([next_state], device=self.device, dtype = torch.float)

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

            if len(self.memory) >= self.STATS_BATCH_SIZE:
                transitions = self.memory.sample(self.STATS_BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.policy_net(state_batch, action_batch)
                dset_q[i_episode] = state_action_values.detach().numpy().mean()

                if self.simple_test_flag:
                    good_actions = []
                    bad_actions = []
                    for sample in transitions:
                        state_diff = sample.state - self.target_assembly
                        good_action = [0,0,0,0]
                        bad_action = [0,0,0,0]
                        if state_diff[0] == 0.0 and state_diff[1] == 0.0:
                            good_action = [0,0,0,1]
                            bad_action = [0,0,0,0]
                        else:
                            if np.absolute(state_diff[0]) >= np.absolute(state_diff[1]):
                                if state_diff[0] > 0.0:
                                    good_action = [1,0,0,0]
                                    bad_action = [0,0,0,0]
                                else:
                                    good_action = [0,0,0,0]
                                    bad_action = [1,0,0,0]
                            else:
                                if state_diff[1] > 0.0:
                                    good_action = [0,0,1,0]
                                    bad_action = [0,1,0,0]
                                else:
                                    good_action = [0,1,0,0]
                                    bad_action = [0,0,1,0]
                        good_action_tensor = torch.tensor(good_action, device = self.device, dtype = torch.float)
                        bad_action_tensor = torch.tensor(bad_action, device = self.device, dtype = torch.float)
                        good_actions.append(good_action_tensor)
                        bad_actions.append(bad_action_tensor)

                    good_action_batch = torch.cat(good_actions)
                    state_good_action_values = self.policy_net(state_batch, good_action_batch)
                    dset_goodq[i_episode] = state_good_action_values.detach().numpy().mean()

                    bad_action_batch = torch.cat(bad_actions)
                    state_bad_action_values = self.policy_net(state_batch, bad_action_batch)
                    dset_badq[i_episode] = state_bad_action_values.detach().numpy().mean()

            else:
                dset_q[i_episode] = 0.0

            dset_rewards[i_episode] = r_episode

            dset_q.flush()
            dset_rewards.flush()

        pickle.dump(self.final_result_per_episode, open( "episode_rewards.p", "wb" ))

        # print('Complete')

if __name__ == "__main__":
    worldsize = [8000, 8000]

    type_infos = [
        # id, radius, propensity
        ["jeb", 1000, 0.5],
        # ["shiet", 1200, 420],
        # ["goteem", 1300, 420]
    ]

    type_counts = [1]
    # type_counts = [1, 1, 1]

    lj_corr_matrix = [[(np.random.random(), np.random.random())]]

    # lj_corr_matrix = [
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())]
    # ]

    target_assembly = np.array([[4000, 4000]])

    cs = ColloidalSystem(worldsize,
                         type_infos,
                         type_counts,
                         lj_corr_matrix,
                         target_assembly)
    cs.set_temperature(300)

    agent = DQNAgent(cs)
    agent.train_model()
