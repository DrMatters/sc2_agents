import logging
import os
import random
from .agent_brain import Agent
from typing import List

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np
from smac.env import StarCraft2Env

# %%

SC2_PATH = '/Applications/StarCraft II'
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10
N_EPISODE = 4000

random.seed(42)
np.random.seed(42)
os.environ['SC2PATH'] = SC2_PATH


def main():
    timesteps = 800000
    learn_freq = 1
    num_exploration = int(timesteps * 0.1)
    eps_decay_steps = timesteps - num_exploration

    env = StarCraft2Env(map_name="2m2zFOX", seed=42, reward_only_positive=False,
                        obs_timestep_number=True, reward_scale_rate=200)
    agents: List[Agent] = prepare_agents(env, eps_decay_steps)
    n_agents = len(agents)
    step = 0
    attack_target_action_offset = 6

    for episode in range(N_EPISODE):
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
                selected_action = agents[agent_id].select_action(observation_before[agent_id])
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
                    print(f"steps until now : {step},"
                          f" episode: {episode},"
                          f" episode reward: {episode_reward_all}")
                    break

                step += 1

                if step == num_exploration:
                    print("Exploration finished")

                if (step > num_exploration) and (step % learn_freq == 0):
                    for agent_id in range(n_agents):
                        agents[agent_id].learn()
                    training_step += 1


def prepare_agents(env: StarCraft2Env, eps_decay_steps):
    env_info = env.get_env_info()
    n_agents = env_info['n_agents']
    agents: List[Agent] = []
    n_actions = env_info['n_actions']
    n_features = env.get_obs_size()
    for i in range(n_agents):
        agents.append(Agent(n_features, n_actions, eps_decay_steps))
    return agents


if __name__ == '__main__':
    main()
