import abc
from typing import Any

import numpy as np
import smac.env as sm_env

from source import individuals


class BaseSCEvaluator(abc.ABC):
    def __init__(self, environment: sm_env.StarCraft2Env):
        self.env = environment
        env_info = self.env.get_env_info()
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

    @abc.abstractmethod
    def evaluate(self, individual: individuals.BaseIndividual) -> Any:
        pass


class SCAbsPosEvaluator(BaseSCEvaluator):
    def evaluate(self, individual: individuals.BaseIndividual) -> float:
        self.env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            agents_states = {}
            actions = np.zeros(self.n_agents, dtype=int)

            # get actions for all agents
            for agent_id in range(self.n_agents):
                agent_info = self.env.get_unit_by_id(agent_id)
                agents_states[agent_id] = self._get_state_fox(agent_info.pos.x, agent_info.pos.y)
                avail_actions = self.env.get_avail_agent_actions(agent_id)
                # avail_actions = [0, 1, 1, 1, 1, 1, 0, 0, 0]
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # avail_actions_ind = [1, 2, 3, 4, 5]
                actions[agent_id] = individual.get_action(agents_states[agent_id], avail_actions_ind)

            reward, terminated, _ = self.env.step(actions)
            episode_reward += reward
        return episode_reward

    @staticmethod
    def _get_state_fox(agent_pos_x, agent_pos_y):
        state = 11  # начальная позиция для случая 1 агента! для 2 может быть ошибка!!
        # print (agent_posX)
        # print (agent_posY)

        if 6 < agent_pos_x < 7 and 16.2 < agent_pos_y < 17:
            state = 0
        elif 7 < agent_pos_x < 8 and 16.2 < agent_pos_y < 17:
            state = 1
        elif 8 < agent_pos_x < 8.9 and 16.2 < agent_pos_y < 17:
            state = 2
        elif 8.9 < agent_pos_x < 9.1 and 16.2 < agent_pos_y < 17:
            state = 3
        elif 9.1 < agent_pos_x < 10 and 16.2 < agent_pos_y < 17:
            state = 4
        elif 10 < agent_pos_x < 11 and 16.2 < agent_pos_y < 17:
            state = 5
        elif 11 < agent_pos_x < 12 and 16.2 < agent_pos_y < 17:
            state = 6
        elif 12 < agent_pos_x < 13.1 and 16.2 < agent_pos_y < 17:
            state = 7
        elif 6 < agent_pos_x < 7 and 15.9 < agent_pos_y < 16.2:
            state = 8
        elif 7 < agent_pos_x < 8 and 15.9 < agent_pos_y < 16.2:
            state = 9
        elif 8 < agent_pos_x < 8.9 and 15.9 < agent_pos_y < 16.2:
            state = 10
        elif 8.9 < agent_pos_x < 9.1 and 15.9 < agent_pos_y < 16.2:
            state = 11
        elif 9.1 < agent_pos_x < 10 and 15.9 < agent_pos_y < 16.2:
            state = 12
        elif 10 < agent_pos_x < 11 and 15.9 < agent_pos_y < 16.2:
            state = 13
        elif 11 < agent_pos_x < 12 and 15.9 < agent_pos_y < 16.2:
            state = 14
        elif 12 < agent_pos_x < 13.1 and 15.9 < agent_pos_y < 16.2:
            state = 15
        elif 6 < agent_pos_x < 7 and 15 < agent_pos_y < 15.9:
            state = 16
        elif 7 < agent_pos_x < 8 and 15 < agent_pos_y < 15.9:
            state = 17
        elif 8 < agent_pos_x < 8.9 and 15 < agent_pos_y < 15.9:
            state = 18
        elif 8.9 < agent_pos_x < 9.1 and 15 < agent_pos_y < 15.9:
            state = 19
        elif 9.1 < agent_pos_x < 10 and 15 < agent_pos_y < 15.9:
            state = 20
        elif 10 < agent_pos_x < 11 and 15 < agent_pos_y < 15.9:
            state = 21
        elif 11 < agent_pos_x < 12 and 15 < agent_pos_y < 15.9:
            state = 22
        elif 12 < agent_pos_x < 13.1 and 15 < agent_pos_y < 15.9:
            state = 23
        elif 6 < agent_pos_x < 7 and 14 < agent_pos_y < 15:
            state = 24
        elif 7 < agent_pos_x < 8 and 14 < agent_pos_y < 15:
            state = 25
        elif 8 < agent_pos_x < 8.9 and 14 < agent_pos_y < 15:
            state = 26
        elif 8.9 < agent_pos_x < 9.1 and 14 < agent_pos_y < 15:
            state = 27
        elif 9.1 < agent_pos_x < 10 and 14 < agent_pos_y < 15:
            state = 28
        elif 10 < agent_pos_x < 11 and 14 < agent_pos_y < 15:
            state = 29
        elif 11 < agent_pos_x < 12 and 14 < agent_pos_y < 15:
            state = 30
        elif 12 < agent_pos_x < 13.1 and 14 < agent_pos_y < 15:
            state = 31

        return state
