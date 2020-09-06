import math
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from smac.env import StarCraft2Env
from torch import nn

# %%

SC2_PATH = 'D:\\Prog\\SC2_reinforcement_learning\\StarCraft II'
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
N_EPISODE = 4000

random.seed(42)
np.random.seed(42)
os.environ['SC2PATH'] = SC2_PATH


# %%


class AgentDQN(nn.Module):
    def __init__(self, n_features, n_actions, hidden_layer_nodes: int = None):
        super(AgentDQN, self).__init__()
        if hidden_layer_nodes is None:
            hidden_layer_nodes = (n_features + n_actions) // 2
        self.layer_1 = nn.Linear(n_features, hidden_layer_nodes)
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(n_features, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


class Agent:
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 memory_size=500,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 batch_size=32):
        self.net = AgentDQN(n_features, n_actions)
        self.n_actions = n_actions
        self.n_features = n_features
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.steps_passed = 0

    def store_transition(self, observation_before, action, reward, observation_after):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((observation_before, [action, reward], observation_after))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def select_action(self, state):
        sample = random.random()
        eps_threshold = \
            self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_passed / self.eps_decay)
        self.steps_passed += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def learn(self):
        if self.memory_counter < self.batch_size:
            return


def main():
    timesteps = 800000
    learn_freq = 1
    num_exploration = int(timesteps * 0.1)
    num_training = timesteps - num_exploration

    env = StarCraft2Env(map_name="2m2zFOX", seed=42, reward_only_positive=False,
                        obs_timestep_number=True, reward_scale_rate=200)
    agents: List[Agent] = prepare_agents(env, num_training)
    n_agents = len(agents)
    step = 0
    attack_target_action_offset = 6

    for episode in range(N_EPISODE):
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0] * n_agents
        reward_health_agent_old = [0] * n_agents
        observation_before = [0] * n_agents

        for agent_id in range(n_agents):
            obs = env.get_obs_agent(agent_id)
            observation_before[agent_id] = obs
            reward_health_agent_old[agent_id] = env.get_agent_health(agent_id)

        while True:
            # RL choose action based on local observation
            selected_actions = [None] * n_agents
            taken_action = [1] * n_agents
            dead_units = set()
            for agent_id in range(n_agents):
                selected_action = agents[agent_id].select_action(observation_before[agent_id])
                selected_actions[agent_id] = selected_action
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if selected_action in avail_actions_ind:
                    taken_action[agent_id] = selected_action
                elif avail_actions[0] == 1:
                    taken_action[agent_id] = 0  # if dead use stub action
                # else: 1 (stop) by default
                # if not dead use 'stop' by default

                if len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0:  # "something" and dead
                    dead_units.add(agent_id)

            # RL take action and get next observation and reward
            env_reward, done, _ = env.step(taken_action)
            episode_reward_all += env_reward
            observation_after = [0] * n_agents
            reward_health_agent_new = [0] * n_agents

            for agent_id in range(n_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                observation_after[agent_id] = obs_next
                reward_health_agent_new[agent_id] = env.get_agent_health(agent_id)

            # obtain proper reward of every agent and store it in transition
            for agent_id in range(n_agents):
                if taken_action[agent_id] > 5:
                    # target_id = taken_action[agent_id] - attack_target_action_offset
                    # health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    # if (health_reduce_en > 0):
                    if env_reward > 0:
                        reward = 2 + env_reward
                    else:
                        reward = 2
                    # else:
                    #     reward = 1
                else:
                    reward = (reward_health_agent_new[agent_id] -
                              reward_health_agent_old[agent_id]) * 5

                if agent_id in dead_units:
                    reward = 0

                episode_reward_agent[agent_id] += reward

                if taken_action[agent_id] == selected_actions[agent_id]:
                    # Save the transition only when the calculated action is the
                    # same as the action taken
                    agents[agent_id].store_transition(
                        observation_before[agent_id],
                        selected_actions[agent_id],
                        reward,
                        observation_after[agent_id]
                    )

                # swap observation (not needed, as observation is
                # received from environment)
                # observation_before = observation_after
                # reward_health_agent_old = reward_health_agent_new
                # reward_hl_en_old = reward_hl_en_new

                # break while loop when end of this episode
                if done:
                    # for i in range(n_agents):
                    #     agents[i].get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)
                    print(
                        f"steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward_all))
                    break

                step += 1

                if step == num_exploration:
                    print("Exploration finished")

                if (step > num_exploration) and (step % learn_freq == 0):
                    for agent_id in range(n_agents):
                        agents[agent_id].learn()
                    training_step += 1


def prepare_agents(env, num_training):
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    agents: List[Agent] = []
    n_actions = env_info['n_actions']
    n_features = env.get_obs_size()
    for i in range(n_agents):
        agents.append(Agent(n_features, n_actions, num_training))
    return agents


if __name__ == '__main__':
    main()
