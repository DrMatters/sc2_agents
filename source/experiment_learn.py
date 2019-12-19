import random
from datetime import datetime

import numpy as np
from deap import algorithms, base, creator, tools
from smac.env import StarCraft2Env

from source import evaluate, individuals

POPULATION = 10
NUM_GENERATIONS = 10
EVALUATE_TOP = True


def main():
    toolbox, evaluator = prepare_env()

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

    if EVALUATE_TOP:
        print("results of evaluation of top individual")
        evaluator.evaluate_single(hof.items[0])

    return pop, stats, hof


def save_top_individual(hof):
    top = hof.items[0]
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    top.save(f'{now}---top_individual_q_table.npy')


def prepare_env():
    random.seed(42)
    np.random.seed(42)
    env = StarCraft2Env(map_name="2m2zFOX", difficulty="1", seed=42)

    evaluator = evaluate.SCAbsPosEvaluator(env)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", individuals.AgentwiseQInd,
                   fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", individuals.AgentwiseQInd.init_simple,
                     creator.Individual, num_agents=evaluator.n_agents,
                     num_states=32, num_actions=evaluator.n_actions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", individuals.AgentwiseQInd.mate_avg)
    toolbox.register("mutate", individuals.AgentwiseQInd.mutate, loc=1,
                     scale=0.05, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox, evaluator


if __name__ == "__main__":
    main()
