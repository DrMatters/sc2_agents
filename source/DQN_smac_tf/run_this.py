from smac.env import StarCraft2Env
from source.DQN_smac_tf.RL_brain import DeepQNetwork
import numpy as np
import tensorflow as tf

def run_this(agents_models, n_episode, learn_freq, n_exploration_episodes, n_agents, ratio_total_reward):
    step = 0
    training_step = 0
    n_actions_no_attack = 6
    for episode in range(n_episode):
        # initial observation
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0] * n_agents
        observation_set = []
        reward_hl_own_old = []
        reward_hl_en_old = []
        for agent_id in range(n_agents):                #第一个循环是为了得到初始状态/观察/生命值信息
            obs = env.get_obs_agent(agent_id)
            observation_set.append(obs)
            reward_hl_own_old.append(env.get_agent_health(agent_id))
            reward_hl_en_old.append(env.get_enemy_health(agent_id))

        while True:
            # RL choose action based on local observation
            action_set_actual = []
            action_set_execute = []
            dead_unit = []
            for agent_id in range(n_agents):
                selected_action = agents_models[agent_id].choose_action(observation_set[agent_id])
                action_set_actual.append(selected_action)
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if selected_action in avail_actions_ind:
                    action_set_execute.append(selected_action)
                elif(avail_actions[0] == 1):
                    action_set_execute.append(0)      #如果该动作不能执行，并且智能体已经死亡，那么就用NO_OP代替当前动作
                else:
                    action_set_execute.append(1)      #如果该动作不能执行，那么就用STOP动作代替

                if (len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):   #判断该智能体是否已经死亡
                    dead_unit.append(agent_id)

            # RL take action and get next observation and reward
            reward_base, done, _ = env.step(action_set_execute)
            episode_reward_all += reward_base
            observation_set_next = []
            reward_hl_own_new = []
            reward_hl_en_new = []

            for agent_id in range(n_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                observation_set_next.append(obs_next)
                reward_hl_own_new.append(env.get_agent_health(agent_id))
                reward_hl_en_new.append(env.get_enemy_health(agent_id))


            # obtain propre reward of every agent and stored it in transition
            for agent_id in range(n_agents):
                if (action_set_execute[agent_id] > 5):
                    target_id = action_set_execute[agent_id] - n_actions_no_attack
                    health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    if(health_reduce_en > 0):
                        if(reward_base > 0):
                            reward = 2 + reward_base
                        else:
                            reward = 2
                    else:
                        reward = 1
                else:
                    reward = (reward_hl_own_new[agent_id] - reward_hl_own_old[agent_id]) * 5

                if(agent_id in dead_unit):
                    reward = 0

                episode_reward_agent[agent_id] += reward

                if(action_set_execute[agent_id] == action_set_actual[agent_id]):     #只有当计算出的动作与所采取的动作一样的时候，才保存下来该transition
                    agents_models[agent_id].store_transition(observation_set[agent_id], action_set_actual[agent_id], reward, observation_set_next[agent_id])

            # swap observation
            observation_set = observation_set_next
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                for i in range(n_agents):
                    agents_models[i].get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward_all))
                break

            step += 1

            if (step == n_exploration_episodes):
                print("Training starts.")

            if (step > n_exploration_episodes) and (step % learn_freq == 0):
                for agent_id in range(n_agents):
                    agents_models[agent_id].learn()
                training_step += 1

            if (training_step >= 10000 and training_step % 10000 == 0):
                print("Model have been trained for %s times" % (training_step))


    # end of game
    print('game over')
    env.close()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="8m", reward_only_positive=False, obs_last_action=True, obs_timestep_number=True,
                        reward_scale_rate=200)  # 8m    reward_scale_rate=200
    env_info = env.get_env_info()

    vector_obs_len = 179  # local observation 80
    n_actions = env_info["n_actions"]
    n_episode = 4000
    n_agents = env_info["n_agents"]
    # episode_len = env_info["episode_limit"]
    learn_freq = 1
    timesteps = 800000
    num_exploration = int(timesteps * 0.1)
    num_training = timesteps - num_exploration
    ratio_total_reward = 0.2

    rl_set = []
    for i in range(n_agents):
        g = tf.Graph()
        sess = tf.Session(graph=g)

        with sess.as_default():
            with g.as_default():

                RL = DeepQNetwork(n_actions=n_actions,
                                  n_features=vector_obs_len,
                                  sess=sess,
                                  agent_id=i,
                                  num_training=num_training,
                                  learning_rate=0.00025,  #0.002
                                  reward_decay=0.99,
                                  replace_target_iter=5000,
                                  memory_size=num_exploration,
                                  batch_size=32,
                                  save_model_freq=10000,
                                  load_model=False,
                                  )

                rl_set.append(RL)

    # run_this写成一个所有智能体执行的函数
    run_this(rl_set, n_episode, learn_freq, num_exploration, n_agents, ratio_total_reward)