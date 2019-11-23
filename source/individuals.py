import abc
from typing import Dict

import numpy as np


class BaseInd(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array]) -> Dict[int, int]:
        pass


class AgentwiseQInd(BaseInd):
    def __init__(self, num_agents, num_states, num_actions, epsilon=0.7):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        self.Q = np.zeros((num_agents, num_states, num_actions), np.float32)

    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array]) -> Dict[int, int]:

        # assume agents ids starts from 0 and goes up to num_agents - 1
        # (including the last one)

        agents_actions = {}
        for agent_id in range(self.num_agents):
            agents_actions[agent_id] = self.get_action(
                agent_id, states[agent_id], avail_actions[agent_id]
            )
        return agents_actions

    def get_action(self, agent_id: int, agent_state: int,
                   avail_actions: np.ndarray) -> np.int64:

        if np.random.rand() < (1 - self.epsilon):
            action = np.random.choice(avail_actions)  # Explore action state
        else:
            actions_val = {}
            for action_index in avail_actions:
                actions_val[action_index] = \
                    self.Q[agent_id, int(agent_state), action_index]
            action = max(actions_val, key=actions_val.get)
        return action
