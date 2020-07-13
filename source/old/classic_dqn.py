import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from smac.env import StarCraft2Env
from torch import nn

# %%

SC2_PATH = 'D:\Distrib\Kirill\Programs\StarCraft II'
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
N_EPISODE = 4000

random.seed(42)
np.random.seed(42)
os.environ['SC2PATH'] = SC2_PATH


# %%

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
    def __init__(self, n_features, n_actions, hidden_layer_nodes: int = None):
        super(AgentDQN, self).__init__()
        hidden_layer_num = (n_features + n_actions) // 2
        self.layer_1 = nn.Linear(n_features, hidden_layer_num)
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(n_features, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


class Agent:
    def __init__(self, n_features: int, n_actions: int, eps_decrease_steps: int, memory_size=500, eps_start=0.9, eps_end=0.05, eps_decay=200):
        self.net = AgentDQN(n_features, n_actions)
        self.n_actions = n_actions
        self.n_features = n_features
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def select_action(self, state, steps_passed):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_passed / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)


def main():
    timesteps = 800000
    num_exploration = int(timesteps * 0.1)
    num_training = timesteps - num_exploration

    env = StarCraft2Env(map_name="2m2zFOX", seed=42, reward_only_positive=False,
                        obs_timestep_number=True, reward_scale_rate=200)
    agents = prepare_agents(env, num_training)
    n_agents = len(agents)

    for episode in range(N_EPISODE):
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0] * n_agents
        observation_set = []
        reward_hl_own_old = []
        reward_hl_en_old = []

        for agent_id in range(n_agents):
            obs = env.get_obs_agent(agent_id)
            observation_set.append(obs)
            reward_hl_own_old.append(env.get_agent_health(agent_id))
            reward_hl_en_old.append(env.get_enemy_health(agent_id))


def prepare_agents(env, num_training):
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    agents = []
    n_actions = env_info['n_actions']
    n_features = env.get_obs_size()
    for i in range(n_agents):
        agents[i] = Agent(n_features, n_actions, num_training)
    return agents


if __name__ == '__main__':
    main()
