import matplotlib
matplotlib.use('Agg')
import argparse
import json
import os.path
import zipfile

import numpy as np
import pandas as pd
from tsmlstarterbot.parsing import parse


def fetch_data_dir(directory, limit):
    """
    Loads up to limit games into Python dictionaries from uncompressed replay files.
    """
    replay_files = sorted([f for f in os.listdir(directory) if
                           os.path.isfile(os.path.join(directory, f)) and f.startswith("replay-")])

    if len(replay_files) == 0:
        raise Exception("Didn't find any game replays. Please call make games.")

    print("Found {} games.".format(len(replay_files)))
    print("Trying to load up to {} games ...".format(limit))

    loaded_games = 0

    all_data = []
    for r in replay_files:
        full_path = os.path.join(directory, r)
        with open(full_path) as game:
            game_data = game.read()
            game_json_data = json.loads(game_data)
            all_data.append(game_json_data)
        loaded_games = loaded_games + 1

        if loaded_games >= limit:
            break

    print("{} games loaded.".format(loaded_games))

    return all_data

def fetch_data_zip(zipfilename, limit):
    """
    Loads up to limit games into Python dictionaries from a zipfile containing uncompressed replay files.
    """
    all_jsons = []
    with zipfile.ZipFile(zipfilename) as z:
        print("Found {} games.".format(len(z.filelist)))
        print("Trying to load up to {} games ...".format(limit))
        for i in z.filelist[:limit]:
            with z.open(i) as f:
                lines = f.readlines()
                assert len(lines) == 1
                d = json.loads(lines[0].decode())
                all_jsons.append(d)
    print("{} games loaded.".format(len(all_jsons)))
    return all_jsons


def main():
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--data", help="Data directory or zip file containing uncompressed games")
    parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default=15)
    parser.add_argument("--seed", type=int, help="Random seed to make the training deterministic")
    parser.add_argument("--bot_to_imitate", help="Name of the bot whose strategy we want to learn")
    parser.add_argument("--dump_features_location", help="Location of hdf file where the features should be stored")
    parser.add_argument("--cache_train_data", help="Location of where to store/read cache data")
    parser.add_argument('--only_cache_data', action='store_true')

    args = parser.parse_args()

    # Make deterministic if needed
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.cache_train_data:
        base_cache_path = os.path.splitext(args.cache_train_data)[0]
        ext = os.path.splitext(args.cache_train_data)[1]
        path_data_input = base_cache_path + '_input' + ext
        path_data_output = base_cache_path + '_output' + ext

    if not args.cache_train_data or not os.path.exists(path_data_input):
        if args.data.endswith('.zip'):
            raw_data = fetch_data_zip(args.data, args.games_limit)
        else:
            raw_data = fetch_data_dir(args.data, args.games_limit)
        # if no cache path or file not exists then calculate train data
        data_input, data_output = parse(raw_data, args.bot_to_imitate)
        if args.cache_train_data:
            np.save(path_data_input, data_input)
            np.save(path_data_output, data_output)
    else:
        print('Reading cache')
        # cache path exists and file exists
        data_input = np.load(path_data_input)
        data_output = np.load(path_data_output)

    if args.only_cache_data:
        return

    data_size = len(data_input)
    training_input, training_output = data_input[:int(0.85 * data_size)], data_output[:int(0.85 * data_size)]
    validation_input, validation_output = data_input[int(0.85 * data_size):], data_output[int(0.85 * data_size):]


    # Save the trained model, so it can be used by the bot
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, os.path.pardir, "models", args.model_name + ".ckpt")
    print("Training finished, serializing model to {}".format(model_path))
    print("Model serialized")

if __name__ == "__main__":
    main()
