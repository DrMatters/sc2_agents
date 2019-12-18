import os
import random

import numpy as np
from smac.env import StarCraft2Env

from source import evaluate, individuals

NUM_EVAL = 20


def load_latest_q_table(location='./'):
    fnames = []
    for file in os.listdir(location):
        if file.endswith('.npy'):
            fnames.append(file)
    return sorted(fnames, reverse=True)


def main():
    random.seed(42)
    np.random.seed(42)
    fnames = load_latest_q_table()
    if fnames:
        top_individual = individuals.AgentwiseQInd.load(fnames[0])
    else:
        raise FileNotFoundError("Found no individuals")

    random.seed(42)
    np.random.seed(42)
    env = StarCraft2Env(map_name="2m2zFOX", difficulty="1", seed=42)
    evaluator = evaluate.SCAbsPosEvaluator(env)

    eval_res = []
    for i in range(NUM_EVAL):
        eval_res.append(evaluator.evaluate(top_individual))

    print(f'Top individual evaluation')
    print(f'min: {np.min(eval_res)}')
    print(f'max: {np.max(eval_res)}')
    print(f'mean: {np.mean(eval_res)}')
    print(f'std: {np.std(eval_res)}')

    return 0


if __name__ == "__main__":
    main()
