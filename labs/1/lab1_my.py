import pickle
import random
import time

import gym
import numpy as np
from tqdm import tqdm

# debug parameters
ENABLE_SLEEP = True
ENABLE_PRINTING = False
RANDOM_SEED = 228
TB_DIR = './tensorboard/'

# constant parameters
TOTAL_EPISODES = 20000
MAX_STEPS = 100
EPSILON_DECAY_RATE = 1 / TOTAL_EPISODES
MIN_EPSILON = 0.001

# rl params
GAMMA = 0.9
LR_RATE = 0.5
INIT_EPS = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def epsilon_greedy(state, epsilon, Q_table, env):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
    return action


def update_q_table(initial_state, result_state, reward, action, Q_table):
    # same as formula for Q-learning
    # Q_new[s, a] <- Q_old[s, a] + alpha * (r + gamma * max_a(Q_old[s_new, a]) - Q_old[s, a])

    Q_table[initial_state, action] = Q_table[initial_state, action] + LR_RATE * \
                                     (reward + GAMMA * np.max(Q_table[result_state, :]) - Q_table[initial_state, action])


def ravel_env_state(env_state, observation_space):
    obs = np.zeros([res.n for res in observation_space.spaces])
    obs[env_state[0], env_state[1], int(env_state[2])] = 1
    obs = np.argwhere(obs.ravel()).item()
    return obs


def main():
    # writer = tensorboardX.SummaryWriter(TB_DIR, 'lab1_blackjack')
    # loading Frozen Lake environment from gym
    env = gym.make('Blackjack-v0')
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    print(f"\nLearning in {str(env.spec)[8:-1]}...\n")

    epsilon = INIT_EPS

    # Q-table initialization
    observation_space_size = 1
    for space in env.observation_space.spaces:
        observation_space_size *= space.n

    Q_table = np.zeros((observation_space_size, env.action_space.n))

    # Start
    for episode in tqdm(range(TOTAL_EPISODES)):
        env_state = env.reset()
        state = ravel_env_state(env_state, env.observation_space)
        step = 0

        # decreasing epsilon
        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY_RATE
        else:
            epsilon = MIN_EPSILON

        # loop within episode
        while step < MAX_STEPS:
            action = epsilon_greedy(state, epsilon, Q_table, env)
            new_state, reward, done, info = env.step(action)
            new_state = ravel_env_state(new_state, env.observation_space)
            # writer.add_scalar('reward', reward, episode)

            # debug to see what's happening ("Ctrl + /" - uncomment highlighted code)

            # print("state --action--> new_state")
            # print("  {}       {}         {}".format(state, action, new_state))
            # env.render()

            # if done and (reward == 0):
            #     reward = -10  # fell into the hole
            # elif done and (reward == 1):
            #     reward = 100  # goal achieved
            # else:
            #     reward = -1  # step on ice

            # doing the learning
            update_q_table(state, new_state, reward, action, Q_table)
            state = new_state
            step += 1

            if done:
                break

    # print("\nQ-table:\n", Q)

    # save Q-table in file on drive (same directory)
    with open("frozenLake_qTable.pkl", 'wb') as f:
        pickle.dump(Q_table, f)

    #################
    # PLAYING STAGE #
    #################

    # comment all below, if you only need to train agent

    print(f"\nPlaying {str(env.spec)[8:-1]}...\n")

    # load q-table from file
    with open("frozenLake_qTable.pkl", 'rb') as f:
        Q_table = pickle.load(f)

    win = 0
    defeat = 0
    draw = 0

    # Evaluation
    for _ in tqdm(range(10000)):
        env_state = env.reset()
        new_state = ravel_env_state(env_state, env.observation_space)
        step = 0

        while step < MAX_STEPS:
            action = np.argmax(Q_table[new_state, :])
            new_state, reward, done, info = env.step(action)
            new_state = ravel_env_state(new_state, env.observation_space)

            # Windows CLI visualization
            if ENABLE_PRINTING:
                try:
                    win_rate = win / (win + defeat) * 100
                    print("Win rate: {}%".format(win_rate))
                    print(f"Wins: {win}\n"
                          f"Defeats: {defeat}\n")
                except ZeroDivisionError:
                    print("Win rate: 0.0%")
                    print(f"Wins: {win}\n"
                          f"Defeats: {defeat}\n")

                env.render()  # show env

                if done:
                    print("\n\tWIN" if reward == 1 else "\n\tDEFEAT")
                    if ENABLE_SLEEP:
                        time.sleep(0.6)

            state = new_state

            if done:
                if reward == 1:
                    win += 1
                elif reward == -1:
                    defeat += 1
                else:
                    draw += 1
                break

            step += 1

        if ENABLE_PRINTING and (step >= MAX_STEPS):
            print("\nTIME IS OVER (steps > 100)")
            if ENABLE_SLEEP:
                time.sleep(1)

    win_rate = win / (win + defeat) * 100
    # writer.close()
    print("Win rate: {}%".format(win_rate))
    print(f'Wins: {win}\nDefeats: {defeat}\nDraw: {draw}')


if __name__ == '__main__':
    main()
