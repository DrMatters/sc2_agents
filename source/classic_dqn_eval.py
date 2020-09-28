import argparse
import logging
import os
import pathlib
import random

import numpy as np
import torch
from smac.env import StarCraft2Env

from source import evaluate
from source import individuals

random.seed(42)
np.random.seed(42)

SEED = 228
# NUM_AGENTS = 2
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='dqn evaluations')
parser.add_argument('-M', '--models_path', default='../results/', type=str,
                    help='path to models * if points to "results" '
                         'selects the latest model')
parser.add_argument('--stored_model_name_mask', default='agent_#.pt',
                    type=str, help='mask to model names for agents')
parser.add_argument('-N', '--number_of_agents', default=2, type=int)


def main():
    args = parser.parse_args()
    env = StarCraft2Env(map_name="15z3m_drm", seed=SEED, reward_only_positive=False,
                        obs_timestep_number=True, reward_scale_rate=200,
                        realtime=False)
    evaluator = evaluate.SCNativeEvaluator(env)

    top_individual = read_last_individual(args, evaluator)

    with torch.no_grad():
        evaluator.evaluate_single(top_individual, 50)
    return 0


def read_last_individual(args, evaluator: evaluate.BaseSCEvaluator):
    models_path = pathlib.Path(args.models_path)

    subdirs = [x for x in models_path.iterdir() if x.is_dir()]
    if len(subdirs):
        latest_models_subdir = max(subdirs, key=os.path.getmtime) / 'models'
        subdirs_models = [x for x in latest_models_subdir.iterdir() if x.is_dir()]
        latest_instance_subdir = max(subdirs_models, key=os.path.getmtime)
        models_path = latest_instance_subdir

    n_agents = args.number_of_agents
    stored_model_name_mask: str = args.stored_model_name_mask
    agents_models = {}
    num_states = evaluator.get_num_states()
    num_actions = evaluator.n_actions

    for agent_id in range(n_agents):
        agent_fname = stored_model_name_mask.replace('#', str(agent_id))
        agent_model = torch.load(models_path / agent_fname)
        agents_models[agent_id] = agent_model.eval()
    top_individual = individuals.AgentwiseFullyConnected(agents_models,
                                                         num_states,
                                                         num_actions)
    return top_individual


if __name__ == '__main__':
    main()
