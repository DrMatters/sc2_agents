import logging
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class AgentDQN(nn.Module):
    def __init__(self, n_features, n_actions, hidden_layer_nodes: int = None):
        super(AgentDQN, self).__init__()
        if hidden_layer_nodes is None:
            hidden_layer_nodes = (n_features + n_actions) // 2
        self.layer_1 = nn.Linear(n_features, hidden_layer_nodes)
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(hidden_layer_nodes, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


class Agent:
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 num_training: int,  # number of calls of 'learn' to
                                     # decrease epsilon to zero
                 memory_size=500,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 replace_target_iter=300,
                 max_epsilon=1.,
                 min_epsilon=0.,
                 batch_size=32):
        self.net = AgentDQN(n_features, n_actions)
        self.n_features = n_features
        self.n_actions = n_actions
        self.num_training = num_training
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replace_target_iter = 300
        self.batch_size = batch_size
        self.memory_counter = 0
        self.steps_passed = 0
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon

    def store_transition(self, observation_before, action, reward, observation_after):
        transition = np.hstack((observation_before, [action, reward], observation_after))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def select_action(self, state):
        # sample = random.random()
        sample = 0.99  # for debug reasons
        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.Tensor(state)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                response = self.net(state)
                # argmax
                return response.max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def learn(self):
        if self.memory_counter < self.batch_size:
            logging.info('Less observation memorized than batch size')
            return
        if self.learn_step_counter % self.replace_target_iter == 0:
            pass

        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.max_epsilon / self.num_training
        else:
            self.epsilon = self.min_epsilon
