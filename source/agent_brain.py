import copy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

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
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


class Agent:
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 eps_decay_steps: int,  # number of calls of 'learn' to
                 # decrease epsilon to zero
                 memory_size=50_000,
                 eps_start=0.9,
                 eps_end=0.05,
                 update_target_every=2,
                 batch_size=10,
                 discount=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        self.eps_decay_steps = eps_decay_steps
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.replace_target_after = 300
        self.batch_size = batch_size
        self.discount = discount
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = self.eps_start
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.model = AgentDQN(n_features, n_actions)
        self.target_model = AgentDQN(n_features)
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam()

    def store_transition(self, observation_before, action, reward, done, observation_after):
        transition = np.hstack((observation_before, [action, reward, done], observation_after))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def select_action(self, state):
        sample = random.random()
        # sample = 0.99  # for debug reasons TODO: replace with proper
        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.Tensor(state)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                response = self.model(state)
                # argmax
                result = response.max(0)[1].view(1, 1)
        else:
            result = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
        return result.cpu().item()

    def learn(self):
        if self.memory_counter < self.batch_size:
            logging.info("There's less observations recorded than batch size")
            return

        if self.learn_step_counter % self.replace_target_after == self.replace_target_after - 1:
            self.target_model.load_state_dict(self.model.state_dict())

            # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Get current states from minibatch, then query NN model for Q values
        # batch contains: idx, (observation_before, [action, reward, done], observation_after)
        current_states = torch.from_numpy(batch_memory[:, :self.n_features]).type(torch.FloatTensor)
        current_qs = self.model(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        future_states = torch.from_numpy(batch_memory[:, -self.n_features:]).type(torch.FloatTensor)
        future_qs = self.target_model(future_states)

        qs_target = current_qs.detach().clone()
        reward = batch_memory[:, self.n_features + 1]

        actions_ind = batch_memory[:, self.n_features].astype(int)
        expect_future_q = self.discount * torch.max(future_qs, dim=1)[0]
        qs_target[:, actions_ind] = \
            torch.from_numpy(reward).type(torch.FloatTensor) + expect_future_q



        self.learn_step_counter += 1

        # update epsilon (exploration probability)
        eps_delta = (self.eps_start - self.eps_end) / self.eps_decay_steps
        eps = max(self.eps_start - eps_delta * self.learn_step_counter,
                  self.eps_end)
        self.epsilon = eps
