import abc
from typing import Dict, Tuple

import numpy as np


class BaseInd(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array],
                    epsilon: float = 0.7) -> Dict[int, int]:
        pass


class BaseGeneticInd(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def initialize_new() -> 'BaseGeneticInd':
        pass

    @staticmethod
    @abc.abstractmethod
    def mate(left: 'BaseGeneticInd', right: 'BaseGeneticInd') \
            -> Tuple['BaseGeneticInd', 'BaseGeneticInd']:
        pass

    @staticmethod
    @abc.abstractmethod
    def mutate(ind):
        pass


class AgentwiseQInd(BaseInd, BaseGeneticInd):
    def __init__(self, num_agents: int, num_states: int, num_actions: int,
                 allocate_q_table: bool = True):
        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions
        if allocate_q_table:
            self.q_table = np.zeros((num_agents, num_states, num_actions),
                                    np.float32)

    @staticmethod
    def from_q_table(q_table):
        ind = AgentwiseQInd(q_table.shape[0], q_table.shape[1],
                            q_table.shape[2], False)
        ind.q_table = q_table
        return ind

    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array],
                    epsilon: float = 0.7) -> Dict[int, int]:

        # assume agents ids starts from 0 and goes up to num_agents - 1
        # (including the last one)

        agents_actions = {}
        for agent_id in range(self.num_agents):
            agents_actions[agent_id] = self._get_action(
                agent_id, states[agent_id], avail_actions[agent_id]
            )
        return agents_actions

    @staticmethod
    def initialize_new(num_agents, num_states, num_actions) -> 'BaseGeneticInd':
        q_table = np.random.random((num_agents, num_states, num_actions))
        return AgentwiseQInd.from_q_table(q_table)

    @staticmethod
    def mate(left: 'AgentwiseQInd', right: 'AgentwiseQInd') \
            -> Tuple['AgentwiseQInd', 'AgentwiseQInd']:

        left_child_q_table = AgentwiseQInd._get_child(left, right)
        right_child_q_table = AgentwiseQInd._get_child(left, right)
        left_child = AgentwiseQInd.from_q_table(left_child_q_table)
        right_child = AgentwiseQInd.from_q_table(right_child_q_table)
        return left_child, right_child

    @staticmethod
    def mutate(ind: 'AgentwiseQInd', loc: float, scale: float,
               indpb: float) -> 'AgentwiseQInd':
        mutate_mask = np.random.random(ind.q_table.shape) < indpb
        mutate_level = np.random.normal(loc, scale, ind.q_table.shape)
        mutate_level_masked = np.multiply(mutate_mask, mutate_level)
        # mutate_level_masked example:
        # here  [[0, 1.28]
        #        [0.97, 0]]
        # we replace all 0s with 1, so when multiply they remain unchanged
        mutate_level_masked[mutate_level_masked == 0] = 1
        mutated_q_table = np.multiply(ind.q_table, mutate_level_masked)
        return AgentwiseQInd.from_q_table(mutated_q_table)

    @staticmethod
    def _get_child(left, right):
        child_q = np.zeros(left.q_table.shape)
        for agent_id in range(left.num_agents):
            agent_q_table_from = AgentwiseQInd._get_rand_q_table_from()
            if agent_q_table_from == 'left':
                child_q[agent_id] = np.copy(left.q_table[agent_id])
            elif agent_q_table_from == 'right':
                child_q[agent_id] = np.copy(right.q_table[agent_id])
            else:
                # array of [1, 0] with the shape of q table of a single agent
                elems_from_left = np.random.randint(2,
                                                    size=left.q_table.shape)
                child_q[agent_id] = np.where(elems_from_left,
                                             left.q_table,
                                             right.q_table)
        return child_q

    @staticmethod
    def _get_rand_q_table_from():
        uncertainty = np.random.random()
        if uncertainty < 0.5:
            q_table_from = 'both'
        elif uncertainty < 0.75:
            q_table_from = 'left'
        else:
            q_table_from = 'right'
        return q_table_from

    def _get_action(self, agent_id: int, agent_state: int,
                    avail_actions: np.ndarray, epsilon: float = 0.7) -> np.int64:

        if np.random.rand() < (1 - epsilon):
            action = np.random.choice(avail_actions)  # Explore action state
        else:
            actions_val = {}
            for action_index in avail_actions:
                actions_val[action_index] = \
                    self.q_table[agent_id, int(agent_state), action_index]
            action = max(actions_val, key=actions_val.get)
        return action
