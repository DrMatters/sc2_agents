import abc
from typing import Dict, Tuple

import numpy as np
import torch

from .agent_brain import AgentDQN


class BaseInd(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array],
                    epsilon: float = 0.7) -> np.ndarray:
        pass


class BaseGeneticInd(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def init_simple(ind_class, num_agents, num_states, num_actions) -> 'BaseGeneticInd':
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


class AgentwiseFullyConnected(BaseInd, BaseGeneticInd):
    def __init__(self, models: Dict[int, AgentDQN], num_states: int, num_actions: int):
        self.models = models
        self.num_agents = len(self.models)
        self.num_states = num_states
        self.num_actions = num_actions

    @staticmethod
    def init_simple(ind_class: BaseInd, num_agents: int, num_states: int, num_actions: int) -> 'BaseGeneticInd':
        # models = [None] * num_agents
        models = {}
        for agent_id in range(num_agents):
            models[agent_id] = AgentDQN(num_states, num_actions).requires_grad_(False)
        return AgentwiseFullyConnected(models, num_states, num_actions)

    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array],
                    epsilon: float = 0.7) -> np.ndarray:
        agents_actions = np.ones(self.num_agents)
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                state = torch.Tensor(states[agent_id])
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                response = self.models[agent_id](state)
                result = response.max(0)[1].view(1, 1).cpu().item()
                if result in avail_actions[agent_id]:  # if action is possible
                    # else remains one (stop)
                    agents_actions[agent_id] = result
        return agents_actions

    @staticmethod
    def mate(left: 'AgentwiseFullyConnected', right: 'AgentwiseFullyConnected') \
            -> Tuple['BaseGeneticInd', 'BaseGeneticInd']:
        return AgentwiseFullyConnected.mate_shuffle(left, right)

    @staticmethod
    def mate_shuffle(left: 'AgentwiseFullyConnected', right: 'AgentwiseFullyConnected') \
            -> Tuple['BaseGeneticInd', 'BaseGeneticInd']:
        # # array of [1, 0] with the shape of q table of a single agent
        l_child = AgentwiseFullyConnected._mate_shuffle_single(left, right)
        r_child = AgentwiseFullyConnected._mate_shuffle_single(left, right)
        return l_child, r_child

    @staticmethod
    def _mate_shuffle_single(left, right):
        with torch.no_grad():
            models = {}
            for agent_id in range(left.num_agents):
                child_model = AgentDQN(left.num_states, left.num_actions)
                layer_name: str
                layer_weights_left: torch.Tensor
                layer_weights_right: torch.Tensor
                for (layer_name, layer_weights_left), (_, layer_weights_right) in zip(
                        left.models[agent_id].state_dict().items(),
                        right.models[agent_id].state_dict().items()
                ):
                    weights_copy_mask = torch.randint(0, 2, layer_weights_left.shape, dtype=torch.bool)
                    child_weights = layer_weights_right.detach().clone()
                    child_weights[weights_copy_mask] = layer_weights_left[weights_copy_mask]
                    child_model.state_dict()[layer_name] = child_weights
                models[agent_id] = child_model
            child = AgentwiseFullyConnected(models, left.num_states, left.num_actions)
        return child

    @staticmethod
    def mutate(ind: 'AgentwiseFullyConnected', loc: float, scale: float,
               indpb: float):
        with torch.no_grad():
            for agent_id in range(ind.num_agents):
                layer_name: str
                layer_weights: torch.Tensor
                for layer_name, layer_weights in ind.models[agent_id].state_dict().items():
                    mutate_mask = np.random.random(layer_weights.shape) < indpb
                    mutate_level = np.random.normal(loc, scale, layer_weights.shape)
                    mutate_level_masked = np.multiply(mutate_mask, mutate_level)
                    # mutate_level_masked example:
                    # here  [[0, 1.28]
                    #        [0.97, 0]]
                    # we replace all 0s with 1, so when multiply they remain unchanged
                    mutate_level_masked[mutate_level_masked == 0] = 1
                    mutate_level_masked_t = torch.tensor(mutate_level_masked,
                                                         dtype=layer_weights.dtype,
                                                         device=layer_weights.device)
                    ind.models[layer_name] = layer_weights * mutate_level_masked_t
                    return ind,


class AgentwiseQTable(BaseInd, BaseGeneticInd):
    def __init__(self, q_table: np.ndarray):
        self.num_agents = q_table.shape[0]
        self.num_states = q_table.shape[1]
        self.num_actions = q_table.shape[2]
        self.q_table = q_table

    def get_actions(self, states: Dict[int, int],
                    avail_actions: Dict[int, np.array],
                    epsilon: float = 0.7) -> np.ndarray:

        # assume agents ids starts from 0 and goes up to num_agents - 1
        # (including the last one)

        agents_actions = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            agents_actions[agent_id] = self._get_action(
                agent_id, states[agent_id], avail_actions[agent_id]
            )
        return agents_actions

    @staticmethod
    def init_simple(ind_class, num_agents, num_states, num_actions):
        q_table = np.random.random((num_agents, num_states, num_actions))
        return ind_class(q_table)

    @staticmethod
    def mate_replace(left: 'AgentwiseQTable', right: 'AgentwiseQTable') \
            -> Tuple['AgentwiseQTable', 'AgentwiseQTable']:

        left_child_q_table = AgentwiseQTable._get_child_rand_replace(left, right)
        right_child_q_table = AgentwiseQTable._get_child_rand_replace(left, right)
        left.q_table = left_child_q_table
        right.q_table = right_child_q_table
        return left, right

    @staticmethod
    def mate(left: 'AgentwiseQTable', right: 'AgentwiseQTable') \
            -> Tuple['AgentwiseQTable', 'AgentwiseQTable']:
        left_child_q_table = AgentwiseQTable._get_child_avg(left, right)
        right_child_q_table = AgentwiseQTable._get_child_avg(left, right)
        left_child = AgentwiseQTable(left_child_q_table)
        right_child = AgentwiseQTable(right_child_q_table)
        return left_child, right_child

    @staticmethod
    def mutate(ind: 'AgentwiseQTable', loc: float, scale: float,
               indpb: float) -> Tuple['AgentwiseQTable']:
        mutate_mask = np.random.random(ind.q_table.shape) < indpb
        mutate_level = np.random.normal(loc, scale, ind.q_table.shape)
        mutate_level_masked = np.multiply(mutate_mask, mutate_level)
        # mutate_level_masked example:
        # here  [[0, 1.28]
        #        [0.97, 0]]
        # we replace all 0s with 1, so when multiply they remain unchanged
        mutate_level_masked[mutate_level_masked == 0] = 1
        mutated_q_table = np.multiply(ind.q_table, mutate_level_masked)
        ind.q_table = mutated_q_table
        return ind,

    def save(self, file):
        np.save(file, self.q_table)

    @staticmethod
    def load(file):
        q_table = np.load(file)
        return AgentwiseQTable(q_table)

    @staticmethod
    def _get_child_rand_replace(left, right):
        child_q = np.zeros(left.q_table.shape)
        for agent_id in range(left.num_agents):
            agent_q_table_from = AgentwiseQTable._get_rand_q_table_from()
            if agent_q_table_from == 'left':
                child_q[agent_id] = np.copy(left.q_table[agent_id])
            elif agent_q_table_from == 'right':
                child_q[agent_id] = np.copy(right.q_table[agent_id])
            else:
                # array of [1, 0] with the shape of q table of a single agent
                elems_from_left = np.random.randint(2, size=left.q_table[agent_id].shape)
                child_q[agent_id] = np.where(elems_from_left,
                                             left.q_table[agent_id],
                                             right.q_table[agent_id])
        return child_q

    @staticmethod
    def _get_child_avg(left, right):
        # child_q = np.copy(left.q_table)
        # left_parent_allele_mask = np.random.random(left.q_table.shape) > 0.5
        # child_q = child_q * left_parent_allele_mask
        #
        # right_parent_allele_mask = left_parent_allele_mask == 0
        # child_q += right.q_table * right_parent_allele_mask
        return (left.q_table + right.q_table) / 2

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
                    avail_actions: np.ndarray) -> np.int64:

        actions_val = {}
        for action_index in avail_actions:
            actions_val[action_index] = \
                self.q_table[agent_id, int(agent_state), action_index]
        return max(actions_val, key=actions_val.get)
