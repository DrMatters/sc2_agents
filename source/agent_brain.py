import logging
import random
from typing import Optional

import numpy as np
import tensorboardX
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
    tb_writer: Optional[tensorboardX.SummaryWriter]

    def __init__(self,
                 agent_id: int,
                 n_features: int,
                 n_actions: int,
                 eps_decay_steps: int,  # number of calls of 'learn' to
                 lr=1e-5,
                 update_target_every_eps=10,
                 # decrease epsilon to zero
                 memory_size=500,
                 eps_start=0.9,
                 eps_end=0.05,
                 batch_size=32,
                 discount=0.9,
                 tb_writer=None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.eps_decay_steps = eps_decay_steps
        self.update_target_every_ep = update_target_every_eps
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.batch_size = batch_size
        self.discount = discount
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = self.eps_start
        self.memory_counter = 0
        self.policy_model = AgentDQN(n_features, n_actions)
        self.target_model = AgentDQN(n_features, n_actions)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.tb_writer = tb_writer

        self.prev_episode_call = -1
        self.tb_prefix = f'agent_no_{agent_id}/'
        self.cost_history = []

    def store_transition(self, observation_before, action, reward, done, observation_after):
        transition = np.hstack((observation_before, [action, reward, done], observation_after))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def select_action(self, state, episode):
        # todo: this is for debug
        if self.tb_writer and episode % 25 == 0:
            self.tb_writer.add_scalar(f'{self.tb_prefix}epsilon', self.epsilon, episode)
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.Tensor(state)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                response = self.policy_model(state)
                # argmax
                result = response.max(0)[1].view(1, 1)
        else:
            result = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
        return result.cpu().item()

    def learn(self, episode):
        # we need to swap models
        if self.prev_episode_call == episode:
            ep_first_call = False
        else:
            ep_first_call = True
        self.prev_episode_call = episode

        if self.memory_counter < self.batch_size:
            logging.info("There's less observations recorded than batch size")
            return

        # swap models
        if episode % self.update_target_every_ep == 0 and ep_first_call:
            logging.info(f'Swapping models, episode {episode}')
            self.target_model.load_state_dict(self.policy_model.state_dict())

        # sample batch memory from memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Get current states from minibatch, then query NN model for Q values
        # batch contains: idx, (observation_before, [action, reward, done], observation_after)
        current_states = torch.from_numpy(batch_memory[:, :self.n_features]).type(torch.FloatTensor)
        current_qs = self.policy_model(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        future_states = torch.from_numpy(batch_memory[:, -self.n_features:]).type(torch.FloatTensor)
        future_qs = self.target_model(future_states)

        # calculate target qs
        qs_target = current_qs.detach().clone()
        reward = batch_memory[:, self.n_features + 1]

        # expected future qs are added to reward
        done = batch_memory[:, self.n_features + 2]
        expect_future_q = self.discount * torch.max(future_qs, dim=1)[0]
        expect_future_q = expect_future_q.detach().cpu().numpy() * (1 - done)

        updated_reward = torch.from_numpy(reward + expect_future_q)
        actions_ind = batch_memory[:, self.n_features].astype(int)
        qs_target[:, actions_ind] = updated_reward.type(torch.FloatTensor)

        # model & optimizer step
        outputs = self.policy_model(current_states)
        loss = self.criterion(outputs, qs_target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.cost_history.append(loss.cpu().item())

        if self.tb_writer:
            self.tb_writer.add_scalar(f'{self.tb_prefix}loss',
                                      loss.item(), episode)

        # update epsilon. Exploration (random move) probability
        eps_delta = (self.eps_start - self.eps_end) / self.eps_decay_steps
        self.epsilon = max(self.eps_start - eps_delta * episode, self.eps_end)
