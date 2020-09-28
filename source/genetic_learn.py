import os
import random
from datetime import datetime
from typing import Type

import numpy as np
import tensorboardX
import torch
from deap import algorithms, base, creator, tools
from smac.env import StarCraft2Env

from source import evaluate, individuals

SEED = 42
POPULATION = 10
NUM_GENERATIONS = 20
EVALUATE_TOP = True
SC2_PATH = 'G:\Programs\StarCraft II'
PRESET = 'dqn'
MAP_NAME = "15z3m_drm"
RESULT_PATH_BASE = f'../results/genetic_{PRESET}_{MAP_NAME}/'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    tb_writer = tensorboardX.SummaryWriter(RESULT_PATH_BASE + 'tensorboard')
    env = StarCraft2Env(map_name=MAP_NAME, seed=SEED,
                        reward_only_positive=False, obs_timestep_number=True,
                        reward_scale_rate=200)

    if PRESET == 'q_table':
        # env = StarCraft2Env(map_name="2m2zFOX", difficulty="1", seed=SEED)
        evaluator = evaluate.SCAbsPosEvaluator(env)
        toolbox = prepare_env(individuals.AgentwiseQTable, evaluator)
    elif PRESET == 'dqn':
        evaluator = evaluate.SCNativeEvaluator(env, tb_writer)
        toolbox = prepare_env(individuals.AgentwiseFullyConnected, evaluator)
    else:
        raise NotImplementedError(f'Preset {PRESET} for genetic learn is not available')

    pop = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=NUM_GENERATIONS,
                        stats=stats, halloffame=hof)
    save_top_individual(hof)
    print(f'Num of evaluations (episodes): {evaluator.evaluation_counter}')

    if EVALUATE_TOP:
        print("results of evaluation of top individual")
        evaluator.evaluate_single(hof.items[0], 0)

    return pop, stats, hof


def save_top_individual(hof):
    top: individuals.BaseInd = hof.items[0]
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    top.save(RESULT_PATH_BASE + f'{now}__top_individual')


def prepare_env(individual_class: Type[individuals.BaseGeneticInd], evaluator):
    os.environ['SC2PATH'] = SC2_PATH

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", individual_class,
                   fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", individual_class.init_simple,
                     creator.Individual, num_agents=evaluator.n_agents,
                     num_states=evaluator.get_num_states(), num_actions=evaluator.n_actions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", individual_class.mate)
    toolbox.register("mutate", individual_class.mutate, loc=1,
                     scale=0.05, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=6)
    return toolbox


if __name__ == "__main__":
    main()
