# -*- coding: utf-8 -*-
"""
https://github.com/oxwhirl/smac

"""
from smac.env import StarCraft2Env
import numpy as np
# import sys
import random
import pickle

# from gym.spaces import Discrete, Box, Dict


# Вывод массива целиком
np.set_printoptions(threshold=np.inf)


# определяем может ли агент сделать заданнное действие action_is
def is_possible_action(avail_actions_ind, action_is):
    ia = 0
    # print ("in def len(avail_actions_ind) = ", len(avail_actions_ind))
    while ia < len(avail_actions_ind):
        # print ("ia = ", ia)
        if avail_actions_ind[ia] == action_is:
            ia = len(avail_actions_ind) + 1
            return True
        else:
            ia = ia + 1

        return False


# получаем состояние агента как позицию на карте
def get_stateFox(agent_posX, agent_posY):
    state = 11
    # print (agent_posX)
    # print (agent_posY)

    if 6 < agent_posX < 7 and 16.2 < agent_posY < 17:
        state = 0
    elif 7 < agent_posX < 8 and 16.2 < agent_posY < 17:
        state = 1
    elif 8 < agent_posX < 8.9 and 16.2 < agent_posY < 17:
        state = 2
    elif 8.9 < agent_posX < 9.1 and 16.2 < agent_posY < 17:
        state = 3
    elif 9.1 < agent_posX < 10 and 16.2 < agent_posY < 17:
        state = 4
    elif 10 < agent_posX < 11 and 16.2 < agent_posY < 17:
        state = 5
    elif 11 < agent_posX < 12 and 16.2 < agent_posY < 17:
        state = 6
    elif 12 < agent_posX < 13.1 and 16.2 < agent_posY < 17:
        state = 7
    elif 6 < agent_posX < 7 and 15.9 < agent_posY < 16.2:
        state = 8
    elif 7 < agent_posX < 8 and 15.9 < agent_posY < 16.2:
        state = 9
    elif 8 < agent_posX < 8.9 and 15.9 < agent_posY < 16.2:
        state = 10
    elif 8.9 < agent_posX < 9.1 and 15.9 < agent_posY < 16.2:
        state = 11
    elif 9.1 < agent_posX < 10 and 15.9 < agent_posY < 16.2:
        state = 12
    elif 10 < agent_posX < 11 and 15.9 < agent_posY < 16.2:
        state = 13
    elif 11 < agent_posX < 12 and 15.9 < agent_posY < 16.2:
        state = 14
    elif 12 < agent_posX < 13.1 and 15.9 < agent_posY < 16.2:
        state = 15
    elif 6 < agent_posX < 7 and 15 < agent_posY < 15.9:
        state = 16
    elif 7 < agent_posX < 8 and 15 < agent_posY < 15.9:
        state = 17
    elif 8 < agent_posX < 8.9 and 15 < agent_posY < 15.9:
        state = 18
    elif 8.9 < agent_posX < 9.1 and 15 < agent_posY < 15.9:
        state = 19
    elif 9.1 < agent_posX < 10 and 15 < agent_posY < 15.9:
        state = 20
    elif 10 < agent_posX < 11 and 15 < agent_posY < 15.9:
        state = 21
    elif 11 < agent_posX < 12 and 15 < agent_posY < 15.9:
        state = 22
    elif 12 < agent_posX < 13.1 and 15 < agent_posY < 15.9:
        state = 23
    elif 6 < agent_posX < 7 and 14 < agent_posY < 15:
        state = 24
    elif 7 < agent_posX < 8 and 14 < agent_posY < 15:
        state = 25
    elif 8 < agent_posX < 8.9 and 14 < agent_posY < 15:
        state = 26
    elif 8.9 < agent_posX < 9.1 and 14 < agent_posY < 15:
        state = 27
    elif 9.1 < agent_posX < 10 and 14 < agent_posY < 15:
        state = 28
    elif 10 < agent_posX < 11 and 14 < agent_posY < 15:
        state = 29
    elif 11 < agent_posX < 12 and 14 < agent_posY < 15:
        state = 30
    elif 12 < agent_posX < 13.1 and 14 < agent_posY < 15:
        state = 31

    return state


"""    
keys = [0 1 2 3 4 5]
act_ind_decode= {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
qt_arr[act_ind]= 0.0
"""


def select_actionFox(state, avail_actions_ind, n_actionsFox, epsilon, Q_table):
    qt_arr = np.zeros(len(avail_actions_ind))
    # Функция arange() возвращает одномерный массив с равномерно разнесенными значениями внутри заданного интервала.
    keys = np.arange(len(avail_actions_ind))
    # print ("keys =", keys)
    # act_ind_decode= {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    # Функция zip объединяет в кортежи элементы из последовательностей переданных в качестве аргументов.
    act_ind_decode = dict(zip(keys, avail_actions_ind))

    stateFoxint = int(state)
    # print ("act_ind_decode=", act_ind_decode)

    for act_ind in range(len(avail_actions_ind)):
        qt_arr[act_ind] = Q_table[stateFoxint, act_ind_decode[act_ind]]
        # print ("qt_arr[act_ind]=",qt_arr[act_ind])

    # Returns the indices of the maximum values along an axis.
    # Exploit learned values
    action = act_ind_decode[np.argmax(qt_arr)]

    return action


# MAIN
def main():
    """The StarCraft II environment for decentralised multi-agent micromanagement scenarios."""
    '''difficulty ="1" is VeryEasy'''
    # replay_dir="D:\StarCraft II\Replays\smacfox"
    env = StarCraft2Env(map_name="2m2mFOX", difficulty="1")

    '''env_info= {'state_shape': 48, 'obs_shape': 30, 'n_actions': 9, 'n_agents': 3, 'episode_limit': 60}'''
    env_info = env.get_env_info()
    # print("env_info = ", env_info)

    """Returns the size of the observation."""
    """obssize =  10"""
    """obs= [array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        0.63521415,  0.63517255, -0.00726997,  0.06666667,  0.06666667],
      dtype=float32)]"""
    obssize = env.get_obs_size()
    # print("obssize = ", obssize)

    ######################################################################
    """
    ready_agents = []
    #observation_space= Dict(action_mask:Box(9,), obs:Box(30,))
    observation_space = Dict({
            "obs": Box(-1, 1, shape=(env.get_obs_size())),
            "action_mask": Box(0, 1, shape=(env.get_total_actions()))  })
    #print ("observation_space=", observation_space)
    
    #action_space= Discrete(9)
    action_space = Discrete(env.get_total_actions())
    #print ("action_space=", action_space)
    """
    ########################################################################

    n_actions = env_info["n_actions"]
    # print ("n_actions=", n_actions)
    n_agents = env_info["n_agents"]

    n_episodes = 10  # количество эпизодов

    ############### Параметры обучения здесь нужны для функции select_actionFox ################################
    alpha = 0.9  # learning rate sayon - 0.5
    gamma = 0.5  # discount factor sayon - 0.9
    epsilon = 0.7  # e-greedy

    n_statesFox = 32  # количество состояний нашего мира-сетки
    n_actionsFox = 7  # вводим свое количество действий, которые понадобятся
    ##################################################################################################
    total_reward = 0

    with open("se17.pkl", 'rb') as f:
        Q_table = pickle.load(f)
        print(Q_table)

    # print (Q_table)

    for e in range(n_episodes):
        # print("n_episode = ", e)
        """Reset the environment. Required after each full episode.Returns initial observations and states."""
        env.reset()
        ''' Battle is over terminated = True'''
        terminated = False
        episode_reward = 0
        actions_history = []
        raz = 0

        # n_steps = 1 #пока не берем это количество шагов для уменьгения награды за долгий поиск

        """
        # вывод в файл
        fileobj = open("файл.txt", "wt")
        print("text",file=fileobj)
        fileobj.close()
        """
        """
        #динамический epsilon
        if e % 15 == 0:
            epsilon += (1 - epsilon) * 10 / n_episodes
            print("epsilon = ", epsilon)
        """

        # stoprun = [0,0,0,0,0]

        while not terminated:
            """Returns observation for agent_id."""
            obs = env.get_obs()
            # print ("obs=", obs)
            """Returns the global state."""
            # state = env.get_state()

            actions = []
            action = 0
            stateFox = np.zeros([n_agents])

            '''agent_id= 0, agent_id= 1'''
            for agent_id in range(n_agents):

                # получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                # получаем состояние по координатам юнита
                stateFox[agent_id] = get_stateFox(unit.pos.x, unit.pos.y)

                """Returns the available actions for agent_id."""
                """avail_actions= [0, 1, 1, 1, 1, 1, 0, 0, 0]"""
                avail_actions = env.get_avail_agent_actions(agent_id)
                '''Функция nonzero() возвращает индексы ненулевых элементов массива.'''
                """avail_actions_ind of agent_id == 0: [1 2 3 4 5]"""
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # выбираем действие
                action = select_actionFox(stateFox[agent_id], avail_actions_ind, n_actionsFox, epsilon, Q_table)
                # собираем действия от разных агентов
                actions.append(action)
                actions_history.append(action)

                ###############_Бежим вправо и стреляем_################################
                """
                if is_possible_action(avail_actions_ind, 6) == True:
                    action = 6
                else:
                    if is_possible_action(avail_actions_ind, 4) == True:
                        action = 4
                    else:
                        action = np.random.choice(avail_actions_ind)
                        #Случайная выборка из значений заданного одномерного массива
                """
                #####################################################################    
                """Функция append() добавляет элементы в конец массива."""
                # print("agent_id=",agent_id,"avail_actions_ind=", avail_actions_ind, "action = ", action, "actions = ", actions)
                # f.write(agent_id)
                # f.write(avail_actions_ind)
                # собираем действия от разных агентов
                # actions.append(action)

            # как узнать куда стрелять? в определенного человека?
            # как узнать что делают другие агенты? самому создавать для них глобальное состояние
            # раз я ими управляю?
            """A single environment step. Returns reward, terminated, info."""
            reward, terminated, _ = env.step(actions)
            episode_reward += reward
            raz = 1

            ###################_Обучаем_##############################################
            """
            for agent_id in range(n_agents):
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox_next = get_stateFox(unit.pos.x, unit.pos.y)
                
            #поменять название на Qlearn
            #подумать над action ведь здесь это последнее действие
            #Qlearn(stateFox, stateFox_next, reward, action)
            
            Q_table[stateFox, action] = Q_table[stateFox, action] + alpha * \
                             (reward + gamma * np.max(Q_table[stateFox_next, :]) - Q_table[stateFox, action])
            """
            ##########################################################################            
        total_reward += episode_reward
        # Total reward in episode 4 = 20.0
        print("Total reward in episode {} = {}".format(e, episode_reward))
        # get_stats()= {'battles_won': 2, 'battles_game': 5, 'battles_draw': 0, 'win_rate': 0.4, 'timeouts': 0, 'restarts': 0}
        print("get_stats()=", env.get_stats())
        print("actions_history=", actions_history)

    # env.save_replay() """Save a replay."""
    print("Average reward = ", total_reward / n_episodes)
    """"Close StarCraft II."""""
    env.close()


if __name__ == "__main__":
    main()
