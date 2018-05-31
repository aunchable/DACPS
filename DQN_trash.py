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
        x = self.fc3(x)
        x = F.softmax(x)

        # TAKE SOFTMAX TO MAKE PROBS OF EACH ACTION SUM TO 1...shouldnt do this in reality
        # x = F.softmax(self.fc3(x))

        # FOR REAL PROBLEM USE SIGMOID
        # x = F.sigmoid(self.fc3(x))

        return x.view(x.size(0), -1)

class DQNAgent():
    def __init__(self, cs):
        self.cs = cs
        self.viz = Visualizer(self.cs)
        self.num_particles = cs.num_particles
        self.state_size = int(3)
        # self.num_actions = int(cs.num_particles)
        self.num_actions = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 10
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 20
        self.TARGET_UPDATE = 10
        self.BUFFER_SIZE = 20
        self.policy_net = DQN(self.num_actions, self.num_particles, self.state_size).to(self.device)
        self.target_net = DQN(self.num_actions, self.num_particles, self.state_size).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.BUFFER_SIZE)
        self.steps_done = 0
        self.num_episodes = 10000
        self.num_time_steps = 150
        self.reward_list = []
        self.final_result_per_episode = []

    # Epsilon-greedy action selection using policy_net
    def select_action(self, state):

        # print("STATE")
        # print(state)
        #
        # print("policy_net")
        # print (self.policy_net(state))

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # print("Best Action")
                # print(self.policy_net(state).max(1)[1].view(1, 1))

                # print("SHIET")
                # print(self.policy_net(state).numpy())
                # print("ANTI SHIET")

                # return torch.tensor((self.policy_net(state).numpy() > 0.5).astype(int), device = self.device, dtype = torch.long )
                return torch.tensor(self.policy_net(state), device = self.device, dtype = torch.float )
                # return self.policy_net(state).max(1)[1].view(1, 1)
        else:

            # print("RAND")
            # return torch.tensor( (np.random.rand(1, self.num_actions) > 0.5).astype(int), device = self.device, dtype = torch.long)
            a = np.random.rand(1, self.num_actions)
            a = a/np.sum(a)
            return torch.tensor(a, device = self.device, dtype = torch.float)
            # return torch.tensor([[random.randint(0, self.num_actions - 1)]], device = self.device, dtype = torch.long)

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

        print("STATE ACTION VALS")
        print(state_action_values)
        print("ACTION BATCH")
        print(action_batch)

        state_action_values = (torch.sum(torch.mul(state_action_values, action_batch.type('torch.FloatTensor') ), dim = 1) ).view(self.BATCH_SIZE, -1)
        # state_action_values_data = state_action_values.data.numpy()
        # action_batch_data = action_batch.numpy()
        # state_action_values_data = np.sum(state_action_values_data * action_batch_data, axis=1)
        # state_action_values = torch.tensor([state_action_values_data], device=self.device, dtype = torch.float).view(self.BATCH_SIZE, -1)

        print("STATE ACTION VALS NEW")
        print(state_action_values)

        # state_action_values = (torch.sum(self.policy_net(state_batch).gather(1, action_batch), dim = 1)).view(self.BATCH_SIZE, -1)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)


        next_state_vals = self.target_net(non_final_next_states)
        print("NEXT STATE VALS")
        print(next_state_vals)

        next_state_vals[next_state_vals < 0] = 0
        print("SHIET")
        print(torch.sum(next_state_vals, dim = 1 ))

        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # max_a' Q^ (psi_new, a' | theta-)
        next_state_values[non_final_mask] = torch.sum(next_state_vals, dim = 1 ) # max_a' Q^ (psi_new, a' | theta-)
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
        for i_episode in range(self.num_episodes):
            print("EPISODE: " + str(i_episode))
            # Initialize the environment and state
            # self.cs.random_initialization()
            self.cs.set_state( np.asarray([[400, 400, 0, 0, 0, 0]])    )
            # self.cs.set_state(np.asarray([[400, 400, 0, 0, 0, 0],[450, 400, 0, 0, 0, 0], [400, 450, 0, 0, 0, 0]]))

            state = self.cs.get_state()[:, :self.state_size]
            # print("STATE")
            # print(state)
            state = [item for sublist in state for item in sublist]
            state = torch.tensor([state], device=self.device, dtype = torch.float)
            r_old = self.cs.get_reward()

            # print("STATE")
            # print(state)

            for t in range(self.num_time_steps):

                # Select and perform an action
                action = self.select_action(state)
                self.steps_done += 1
                # print("ACTION")
                # print(action)
                light_mask = action.numpy().tolist()[0]
                # print("lightmask")
                # print(light_mask)
                # assert(False)
                # light_mask = (( (((action.item() & (1 << np.arange(self.num_particles)))) > 0).astype(int) ).tolist() )

                # Add visualization
                # if t % 1 == 0:
                #     self.viz.update()

                # Do action
                # for j in range(200):
                #     self.cs.step(0.001, light_mask)

                positions = self.cs.state[:, :2]
                thing_to_do = np.argmax(light_mask)
                # print("LIGHT MASK")
                # print(light_mask)

                if thing_to_do == 0:
                    positions[0][0] += 8
                elif thing_to_do == 1:
                    positions[0][0] -= 8
                elif thing_to_do == 2:
                    positions[0][1] += 8
                elif thing_to_do == 3:
                    positions[0][1] -= 8
                else:
                    print("ERROR")
                #
                # # TEMPORARY REMOVE THIS!!!!
                action_vec = np.zeros(4)
                action_vec[thing_to_do] = 1
                action = torch.tensor([action_vec], device = self.device, dtype = torch.long)
                print("ACTION")
                print(action)

                # Add visualization
                # if t % 1 == 0:
                #     time.sleep(0.1)

                # Get reward
                r_new = self.cs.get_reward()
                reward = torch.tensor([r_new - r_old], device=self.device, dtype = torch.float)
                self.reward_list.append(r_new - r_old)

                # if t % 10 == 0:
                #     print("Reward: " + str(r_new - r_old))

                # Observe new state
                next_state = self.cs.get_state()[:, :self.state_size]
                next_state = [item for sublist in next_state for item in sublist]
                next_state = torch.tensor([next_state], device=self.device, dtype = torch.float)

                # Compute TD Error
                # print("ACTION")
                # print(action)
                # TODO: WHEN ACTION IS A BINARY VECTOR MAKE SURE GATHER DOES WHAT YOU THINK IT DOES (SUM OVER Q VALS?)

                print("POLLICY NET OUTPUT")
                print(self.policy_net(state))

                index_to_select = torch.tensor(thing_to_do, device = self.device, dtype = torch.long)

                state_action_val = self.policy_net(state)[0][thing_to_do]
                print("NEW SHIET")
                print(state_action_val)

                print("TARGET NET OUTPUT")
                print(self.target_net(next_state))

                next_state_target_vals = self.target_net(next_state)
                next_state_target_vals[next_state_target_vals < 0] = 0
                print("SHIET")
                print(torch.sum(next_state_target_vals) )

                # max_target_val = self.target_net(next_state).max(1)[0].detach()
                max_target_val = torch.sum(next_state_target_vals)
                print("MAX VAL")
                print(max_target_val)
                # assert(False)
                td_error = max_target_val*self.GAMMA + reward

                # TODO: actually use td error for prioritized sampling from replay buffer

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

                r_old = r_new

            # Show reward at end of episode
            # print("Reward: " + str(self.reward_list[-1]) )

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.final_result_per_episode.append( self.cs.get_reward()  )
            print("Episode Reward: " + str(self.cs.get_reward()))

        pickle.dump(self.final_result_per_episode, open( "episode_rewards.p", "wb" ))

        # print('Complete')

if __name__ == "__main__":
    worldsize = [800, 800]

    type_infos = [
        # id, radius, propensity
        ["jeb", 1000, 0.4],
        # ["shiet", 1200, 420],
        # ["goteem", 1300, 420]
    ]

    # type_counts = [1, 1, 1]
    type_counts = [1]

    lj_corr_matrix = [[(np.random.random(), np.random.random())]]

    # lj_corr_matrix = [
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
    #     [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())]
    # ]

    # target_assembly = np.array([[0, 0], [0, 1], [1, 0]])
    target_assembly = np.array([[0, 0]])

    cs = ColloidalSystem(worldsize,
                         type_infos,
                         type_counts,
                         lj_corr_matrix,
                         target_assembly)
    cs.set_temperature(300)

    agent = DQNAgent(cs)
    agent.train_model()
