import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from simulator.ColloidalSystem import ColloidalSystem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'td_error'))

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

    def __init__(self, num_actions, num_particles):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6*num_particles, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x.view(x.size(0), -1))

# Epsilon-greedy action selection using policy_net
def select_action(policy_net, state, num_actions, device, EPS_START, EPS_END, EPS_DECAY, steps_done):

    print("STATE")
    print(state)

    print("policy_net")
    print (policy_net(state))

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            print("Best Action")
            print(policy_net(state).max(1)[1].view(1, 1))
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, num_actions - 1)]], device = device, dtype = torch.long)

def optimize_model(optimizer, memory, device, BATCH_SIZE, policy_net, target_net, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() # max_a' Q^ (psi_new, a' | theta-)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # gamma max_a' Q^ (psi_new, a' | theta-) + r

    # Compute Huber loss between gamma max_a' Q^ (psi_new, a' | theta-) + r and Q(psi_old, a* | theta)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def run_DQN():
    # --------------------- Initialize colloidal system ---------------------
    worldsize = [800, 800]

    type_infos = [
        # id, radius, propensity
        ["jeb", 10, 420],
        ["shiet", 20, 420],
        ["goteem", 30, 420]
    ]

    type_counts = [1, 1, 1]

    lj_corr_matrix = [
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())],
        [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())]
    ]

    target_assembly = None

    cs = ColloidalSystem(worldsize,
                         type_infos,
                         type_counts,
                         lj_corr_matrix,
                         target_assembly)

    num_particles = cs.num_particles
    num_actions = int(2**(num_particles) )

    # ---------------------------- Setup DQN --------------------------------

    # # set up matplotlib
    # is_ipython = 'inline' in matplotlib.get_backend()
    # if is_ipython:
    #     from IPython import display
    #
    # plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BATCH_SIZE = 128
    BATCH_SIZE = 10
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    BUFFER_SIZE = 20

    policy_net = DQN(num_actions, num_particles).to(device)
    target_net = DQN(num_actions, num_particles).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(BUFFER_SIZE)

    steps_done = 0

    num_episodes = 5
    num_time_steps = 20
    reward_list = []

    # ---------------------------- Run DQN --------------------------------

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        cs.random_initialization()

        state = cs.get_state()
        state = [item for sublist in state for item in sublist]
        state = torch.tensor([state], device=device, dtype = torch.float)

        for t in range(0, num_time_steps):
            # Select and perform an action
            action = select_action(policy_net, state, num_actions, device, EPS_START, EPS_END, EPS_DECAY, steps_done)
            steps_done += 1
            light_mask = ( (((action.item() & (1 << np.arange(num_actions)))) > 0).astype(int) ).tolist()
            cs.step(1, light_mask)

            reward = cs.get_reward()
            reward_list.append(reward)
            reward = torch.tensor([reward], device=device, dtype = torch.float)

            # Observe new state
            next_state = cs.get_state()
            next_state = [item for sublist in next_state for item in sublist]
            next_state = torch.tensor([next_state], device=device, dtype = torch.float)

            # Compute TD Error
            state_action_val = policy_net(state).gather(1, action)
            max_target_val = target_net(next_state).max(1)[0].detach()
            td_error = max_target_val*GAMMA + reward

            # TODO: actually use td error for prioritized sampling from replay buffer

            # Store the transition in memory
            memory.push(state, action, next_state, reward, td_error)
            print("MEMORY")
            print(memory.position)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(optimizer, memory, device, BATCH_SIZE, policy_net, target_net, GAMMA)

        # Show reward at end of episode
        print("Reward: " + str(reward_list[-1]) )

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

run_DQN()
