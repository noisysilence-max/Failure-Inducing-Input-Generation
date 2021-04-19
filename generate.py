

from src.wolp_agent import *
from src.ddpg.agent import DDPGAgent
from cps_env import cps
import src.util.data
from src.util.timer import Timer

import logging


def run(episodes=10,
        experiment='swat',
        max_actions=2**26,
        knn=0.0001,
        in_coeff=0.01,
        ext_coeff=1,
        expl=0.7,
        flag = 1,
        l_rate = 0.0001,
        Discrete = True,
        RND = True,
        i=1):

    env = cps()

    print(env.observation_space)
    print(env.action_space)

    steps = 1000
    if Discrete:
        agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn, l=l_rate)
    else:
        agent = DDPGAgent(env, l_rate=l_rate)

    timer = Timer()

    data = src.util.data.Data(flag)
    data.set_agent(agent.get_name(), int(agent.action_space.get_number_of_actions()),
                   agent.k_nearest_neighbors, 3)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)

    agent.add_data_fetch(data)
    print(data.get_file_name())

    full_epoch_timer = Timer()
    reward_sum = 0

    for ep in range(episodes):
        timer.reset()
        observation = env.reset()
        logging.debug('episodes:{}'.format(ep))

        total_reward = 0
        end = False
        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):
            if np.random.uniform() > expl:  # exploration
                action = np.array(np.random.randint(2, size=agent.action_space_size))
            else:
                action = agent.act(observation)
            data.set__action(np.array([action]).tolist())

            prev_observation = observation
            observation_, ext_reward, done, f = env.step(np.array([action]), prev_observation,flag,t)


            data.set__state(np.array(observation_).tolist())

            if RND :
                in_reward = agent.get_in_reward(observation_)
            else:
                in_reward = 0.0

            data.set_i_reward(in_reward)
            data.set_e_reward(ext_reward)

            reward = in_reward * in_coeff + ext_reward * ext_coeff

            data.set_reward(reward)

            observation = observation_

            episode = {'obs': np.array(prev_observation),
                       'action': np.array(action),
                       'reward': reward,
                       'obs2': np.array(observation),
                       'done': done,
                       'i_reward':in_reward,
                       't': t}

            agent.observe(episode)

            total_reward += reward

            if done:
                end = True

            if done or (t == steps - 1):
                data.set_step(t)
                t += 1
                reward_sum += total_reward
                time_passed = timer.get_time()
                data.set_ep_time(time_passed / 1000)
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,time_passed, round(time_passed / t),np.round(reward_sum / (ep + 1))))
                time = full_epoch_timer.get_time()
                print("time:", time / 1000)
                data.finish_and_store_episode(exp = expl,l_r = l_rate,flag = flag,i=i)
                break
        if end:
            break

    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        ep, time / 1000, reward_sum / (ep + 1)))

    data.save(exp = expl,l_r = l_rate,flag = flag, j=i)

if __name__ == '__main__':
    flag = [1,2,3,4,5,6]            #signal of different failure
    n = 10                       # times of experiment


    for f in flag:
        for i in range(n):
            run(expl=0.7, l_rate=0.001, flag=f, i=i)

