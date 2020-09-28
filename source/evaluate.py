import abc
from typing import Tuple, Any

import numpy as np
import smac.env as sm_env
import tensorboardX

from source import individuals


class BaseSCEvaluator(abc.ABC):
    def __init__(self, environment: sm_env.StarCraft2Env,
                 tb_writer: tensorboardX.SummaryWriter = None, epsilon: float = 0.7):
        self.env = environment
        env_info = self.env.get_env_info()
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]
        self.writer = tb_writer
        self.epsilon = epsilon
        self.evaluation_counter = 0

    @abc.abstractmethod
    def get_num_states(self) -> int:
        pass

    @abc.abstractmethod
    def get_agent_state(self, agent_id: int) -> Any:
        pass

    def get_avail_and_states(self):
        agents_states = {}
        avail_actions_indices = {}
        for agent_id in range(self.n_agents):
            agents_states[agent_id] = self.get_agent_state(agent_id)
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            # avail_actions = [0, 1, 1, 1, 1, 1, 0, 0, 0]
            avail_actions_ind = np.nonzero(avail_actions)[0]
            avail_actions_indices[agent_id] = avail_actions_ind
            # avail_actions_ind = [1, 2, 3, 4, 5]
        return agents_states, avail_actions_indices

    def evaluate(self, individual: individuals.BaseInd) -> Tuple[float]:
        self.env.reset()
        if self.writer and self.env.battles_game:
            win_rate = self.env.get_stats()['win_rate']
            self.writer.add_scalar('win_rate', win_rate, self.evaluation_counter)
        terminated = False
        episode_reward = 0
        while not terminated:
            agents_states, avail_actions_indices = self.get_avail_and_states()
            actions = individual.get_actions(agents_states, avail_actions_indices)

            reward, terminated, _ = self.env.step(actions)
            episode_reward += reward
        self.evaluation_counter += 1
        return episode_reward,

    def evaluate_single(self, individual, n=10):
        eval_res = []
        for i in range(n):
            res = self.evaluate(individual)
            eval_res.append(res)
            print(f'Episode result: {res}')

        print(f'Top individual evaluation')
        print(f'min: {np.min(eval_res)}')
        print(f'max: {np.max(eval_res)}')
        print(f'mean: {np.mean(eval_res)}')
        print(f'std: {np.std(eval_res)}')
        return eval_res


class SCNativeEvaluator(BaseSCEvaluator):
    def get_num_states(self) -> int:
        return self.env.get_obs_size()

    def get_agent_state(self, agent_id):
        return self.env.get_obs_agent(agent_id)

class SCAbsPosEvaluator(BaseSCEvaluator):
    def get_num_states(self) -> int:
        return 32

    def get_agent_state(self, agent_id):
        agent_info = self.env.get_unit_by_id(agent_id)
        agent_state = self._get_state_fox(agent_info.pos.x, agent_info.pos.y)
        return agent_state

    @staticmethod
    def _get_state_fox(agent_pos_x, agent_pos_y):
        state = 11  # начальная позиция для случая 1 агента! для 2 может быть ошибка!!
        # print (agent_posX)
        # print (agent_posY)
        # this is legacy code
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
