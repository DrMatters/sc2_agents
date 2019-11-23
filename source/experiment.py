import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from smac.env import StarCraft2Env

from source import evaluate
from source import individuals

random.seed(42)
np.random.seed(42)
env = StarCraft2Env(map_name="2m2mFOX", difficulty="1")
evaluator = evaluate.SCAbsPosEvaluator(env)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", individuals.AgentwiseQInd, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", individuals.AgentwiseQInd.init_simple,
                 creator.Individual, num_agents=evaluator.n_agents,
                 num_states=32, num_actions=evaluator.n_actions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluator.evaluate)
toolbox.register("mate", individuals.AgentwiseQInd.mate)
toolbox.register("mutate", individuals.AgentwiseQInd.mutate, loc=1,
                 scale=0.05, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, stats=stats,
                        halloffame=hof)

    return pop, stats, hof


if __name__ == "__main__":
    main()
