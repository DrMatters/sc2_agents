import os
import itertools
import random
from collections import namedtuple

import numpy as np
import torch.nn.functional as F
from smac.env import StarCraft2Env
from torch import nn, optim
import torch
import math

from source import evaluate

# %%

SC2_PATH = 'D:\Distrib\Kirill\Programs\StarCraft II'
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

random.seed(42)
np.random.seed(42)
os.environ['SC2PATH'] = SC2_PATH

# %%

env = StarCraft2Env(map_name="2m2zFOX", difficulty="1", seed=42)
evaluator = evaluate.SCAbsPosEvaluator(env)

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


# %%

class AgentDQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(AgentDQN, self).__init__()
        hidden_layer_num = (num_states + num_actions) // 2
        self.layer_1 = nn.Linear(num_states, hidden_layer_num)
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(num_states, num_actions)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


def select_action(state, n_actions, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def model_step(memory, target_net: nn.Module, policy_net: nn.Module):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE,)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# %%
class Agent:
    def __init__(self, policy_net, target_net, optimizer):


def main():
    agents = []
    for i in range(evaluator.n_agents):
        policy_net = AgentDQN(32, env.get_env_info()['n_actions'])
        target_net = AgentDQN(32, env.get_env_info()['n_actions'])
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        agent = {'policy': policy_net, 'target': target_net}
        optimizer = optim.RMSprop(policy_net.parameters())
        agents.append(agent)


    memory = ReplayMemory(10000)
    num_episodes = 50


    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_state = evaluator.get_()
        current_state = get_screen()
        state = current_state - last_state
        for _ in itertools.count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_state = current_state
            current_state = get_screen()
            if not done:
                next_state = current_state - last_state
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            model_step()
            if done:
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()