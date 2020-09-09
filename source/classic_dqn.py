import datetime
import logging
import os
import pathlib
import random
from typing import List

import numpy as np
import tensorboardX
import torch
from smac.env import StarCraft2Env

from .agent_brain import Agent

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# %%

MODEL_NAME = 'basic_inp_avg_out'
SC2_PATH = '/Applications/StarCraft II'
RESULT_PATH_BASE = './results/'
SAVE_FREQ = 100

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
N_EPISODE = 4000

os.environ['SC2PATH'] = SC2_PATH


def main():
    if RESULT_PATH_BASE:
        launch_time = datetime.datetime.now()
        save_path_base = pathlib.Path(RESULT_PATH_BASE)
        save_path_base = save_path_base / f'{MODEL_NAME}' \
                                          f'_d{launch_time:%Y_%m_%d}' \
                                          f'_t{launch_time:%H_%M_%S}'
    else:
        save_path_base = None

    if save_path_base:
        tb_path = save_path_base / 'tensorboard'
        tb_writer = tensorboardX.SummaryWriter(str(tb_path.resolve()))
    else:
        tb_writer = None

    # timesteps = 800000
    timesteps = 100000  # place a proper number here
    learn_freq = 1
    num_exploration = int(timesteps / 10)
    eps_decay_steps = timesteps - num_exploration

    env = StarCraft2Env(map_name="2m2zFOX", seed=42, reward_only_positive=False,
                        obs_timestep_number=True, reward_scale_rate=200)
    agents: List[Agent] = prepare_agents(env, eps_decay_steps)
    n_agents = len(agents)
    step = 0
    # attack_target_action_offset = 6

    for episode in range(N_EPISODE):
        logging.info(f'Episode {episode} has started')
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0] * n_agents
        agent_health_before = [0] * n_agents
        observation_before = [0] * n_agents

        for agent_id in range(n_agents):
            obs = env.get_obs_agent(agent_id)
            observation_before[agent_id] = obs
            agent_health_before[agent_id] = env.get_agent_health(agent_id)

        while True:
            # RL choose action based on local observation
            selected_actions = [None] * n_agents
            taken_actions = [1] * n_agents
            dead_units = set()
            for agent_id in range(n_agents):
                selected_action = agents[agent_id].select_action(
                    observation_before[agent_id], episode)
                selected_actions[agent_id] = selected_action
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if selected_action in avail_actions_ind:
                    taken_actions[agent_id] = selected_action
                elif avail_actions[0] == 1:
                    taken_actions[agent_id] = 0  # if dead use stub action
                # else: 1 (stop) by default
                # if not dead use 'stop' by default

                if len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0:  # "something" and dead
                    dead_units.add(agent_id)

            # RL take action and get next observation and reward
            env_reward, done, _ = env.step(taken_actions)
            episode_reward_all += env_reward
            observation_after = [0] * n_agents
            agent_health_after = [0] * n_agents

            for agent_id in range(n_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                observation_after[agent_id] = obs_next
                agent_health_after[agent_id] = env.get_agent_health(agent_id)

            # obtain proper reward of every agent and store it in transition
            for agent_id in range(n_agents):
                if taken_actions[agent_id] > 5:
                    # target_id = taken_actions[agent_id] - attack_target_action_offset
                    # health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    # if (health_reduce_en > 0):
                    reward = 2 + max(0, env_reward)
                    # else:
                    #     reward = 1
                else:
                    reward = (agent_health_after[agent_id] -
                              agent_health_before[agent_id]) * 5

                if agent_id in dead_units:
                    reward = 0

                episode_reward_agent[agent_id] += reward

                if taken_actions[agent_id] == selected_actions[agent_id]:
                    # Save the transition only when the calculated action is the
                    # same as the action taken
                    agents[agent_id].store_transition(
                        observation_before[agent_id],
                        selected_actions[agent_id],
                        reward,
                        done,
                        observation_after[agent_id]
                    )

            # swap observation (not needed, as observation is
            # received from environment)
            # observation_before = observation_after
            # agent_health_before = agent_health_after
            # reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                # for i in range(n_agents):
                #     agents[i].get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)

                logging.info(f"steps until now : {step},"
                             f" episode: {episode},"
                             f" episode reward: {episode_reward_all}")
                if tb_writer:
                    tb_writer.add_scalar('episode_reward_all', episode_reward_all, episode)
                    for agent_id in range(n_agents):
                        tb_writer.add_scalar(
                            f'agent_no_{agent_id}/episode_reward',
                            episode_reward_agent[agent_id], episode
                        )
                if save_path_base and episode % SAVE_FREQ == SAVE_FREQ - 1:
                    now = datetime.datetime.now()
                    base_model_path = save_path_base / 'models'
                    base_model_path.mkdir(parents=True, exist_ok=True)
                    for agent_id in range(n_agents):
                        model_path = base_model_path / f'd{now:%Y_%m_%d}' \
                                                       f'_t{now:%H_%M_%S}' \
                                                       f'_ep{episode}' \
                                                       f'_agent_no_{agent_id}' \
                                                       f'.torch'
                        logging.info(f'saving model {model_path}')
                        torch.save(agents[agent_id], model_path)
                break

            step += 1

            if step == num_exploration:
                logging.info("Exploration finished")

            if (step >= num_exploration) and (step % learn_freq == 0):
                for agent_id in range(n_agents):
                    agents[agent_id].learn(episode)


def prepare_agents(env: StarCraft2Env, eps_decay_steps):
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    agents: List[Agent] = []
    n_actions = env_info['n_actions']
    n_features = env.get_obs_size()
    for i in range(n_agents):
        agents.append(Agent(i, n_features, n_actions, eps_decay_steps))
    return agents


if __name__ == '__main__':
    main()
