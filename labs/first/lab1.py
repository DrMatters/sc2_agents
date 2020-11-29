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

# constant parameters
TOTAL_EPISODES = 20_000
MAX_STEPS = 100
EPSILON_DECAY_RATE = 1 / TOTAL_EPISODES
MIN_EPSILON = 0.001

# rl params
GAMMA = 0.9
LR_RATE = 0.19
INIT_EPS = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def epsilon_greedy(state, epsilon, Q, env):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update_q_table(initial_state, result_state, reward, action, Q):
    # same as formula for Q-learning
    # Q_new[s, a] <- Q_old[s, a] + alpha * (r + gamma * max_a(Q_old[s_new, a]) - Q_old[s, a])

    Q[initial_state, action] = Q[initial_state, action] + LR_RATE * \
                               (reward + GAMMA * np.max(Q[result_state, :]) - Q[initial_state, action])


def main():
    # loading Frozen Lake environment from gym
    env = gym.make('FrozenLake-v0')
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    print(f"\nLearning in {str(env.spec)[8:-1]}...\n")

    epsilon = INIT_EPS

    # Q-table initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # Start
    for _ in tqdm(range(TOTAL_EPISODES)):
        state = env.reset()
        step = 0

        # decreasing epsilon
        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY_RATE
        else:
            epsilon = MIN_EPSILON

        # loop within episode
        while step < MAX_STEPS:
            action = epsilon_greedy(state, epsilon, Q, env)
            new_state, reward, done, info = env.step(action)

            # debug to see what's happening ("Ctrl + /" - uncomment highlighted code)

            # print("state --action--> new_state")
            # print("  {}       {}         {}".format(state, action, new_state))
            # env.render()

            if done and (reward == 0):
                reward = -10  # fell into the hole
            elif done and (reward == 1):
                reward = 100  # goal achieved
            else:
                reward = -1  # step on ice

            # doing the learning
            update_q_table(state, new_state, reward, action, Q)
            state = new_state
            step += 1

            if done:
                break

    # print("\nQ-table:\n", Q)

    # save Q-table in file on drive (same directory)
    with open("frozenLake_qTable.pkl", 'wb') as f:
        pickle.dump(Q, f)

    #################
    # PLAYING STAGE #
    #################

    # comment all below, if you only need to train agent

    print(f"\nPlaying {str(env.spec)[8:-1]}...\n")

    # load q-table from file
    with open("frozenLake_qTable.pkl", 'rb') as f:
        Q = pickle.load(f)

    win = 0
    defeat = 0

    # Evaluation
    for _ in tqdm(range(1000)):
        state = env.reset()
        step = 0

        while step < MAX_STEPS:
            action = np.argmax(Q[state, :])
            new_state, reward, done, info = env.step(action)

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
                else:
                    defeat += 1
                break

            step += 1

        if ENABLE_PRINTING and (step >= MAX_STEPS):
            print("\nTIME IS OVER (steps > 100)")
            if ENABLE_SLEEP:
                time.sleep(1)

    win_rate = win / (win + defeat) * 100
    print("Win rate: {}%".format(win_rate))


if __name__ == '__main__':
    main()
