import random
import numpy as np
import math as math

from gym.spaces import Box

from src.ddpg.actor_net import ActorNet
from src.ddpg.critic_net import CriticNet
from src.ddpg.actor_net_bn import ActorNet_bn
from src.ddpg.critic_net_bn import CriticNet_bn
from src.ddpg.tensorflow_grad_inverter import grad_inverter
from src.util.target_net import TargetNet
from src.util.prediction_net import PredictNet

from collections import deque



class Agent:

    def __init__(self, env):
        # checking state space
        if isinstance(env.observation_space, Box):
            self.observation_space_size = env.observation_space.shape[1]
        else:
            self.observation_space_size = env.observation_space.n

        # checking action space
        if isinstance(env.action_space, Box):
            self.action_space_size = env.action_space.shape[1]
            self.continious_action_space = True
            self.low = env.action_space.low
            self.high = env.action_space.high
        else:
            self.action_space_size = 1
            self.continious_action_space = False
            self.low = np.array([0])
            self.high = np.array([env.action_space.n])

    def act(self, state):
        pass

    def observe(self, episode):
        pass

    def get_name(self):
        return 'Agent'

    def _np_shaping(self, array, is_state):
        number_of_elements = array.shape[0] if len(array.shape) > 1 else 1
        size_of_element = self.observation_space_size if is_state else self.action_space_size

        res = np.array(array)
        res.shape = (number_of_elements, size_of_element)
        return res



class DDPGAgent(Agent):
    ''' stevenpjg's implementation of DDPG algorithm '''

    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99

    def __init__(self, env, l_rate = 0.0001, is_batch_norm=False, is_grad_inverter=True):
        super().__init__(env)

        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.observation_space_size,
                                           self.action_space_size)
            self.actor_net = ActorNet_bn(self.observation_space_size,
                                         self.action_space_size)

        else:
            self.critic_net = CriticNet(self.observation_space_size,
                                        self.action_space_size,l_rate)
            self.actor_net = ActorNet(self.observation_space_size,
                                      self.action_space_size,l_rate)

        self.target_net = TargetNet(self.observation_space_size)
        self.predict_net = PredictNet(self.observation_space_size)

        self.is_grad_inverter = is_grad_inverter
        self.replay_memory = deque()

        self.time_step = 0

        action_max = np.array(self.high).tolist()
        action_min = np.array(self.low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

    def add_data_fetch(self, df):
        self.data_fetch = df

    def get_name(self):
        return 'DDPG' + super().get_name()

    def act(self, state):
        state = self._np_shaping(state, True)
        result = self.actor_net.evaluate_actor(state).astype(float)
        self.data_fetch.set_actors_action(result[0].tolist())
        return result

    def observe(self, episode):
        episode['obs'] = self._np_shaping(episode['obs'], True)
        episode['action'] = self._np_shaping(episode['action'], False)
        episode['obs2'] = self._np_shaping(episode['obs2'], True)
        self.add_experience(episode)

    def add_experience(self, episode):
        self.replay_memory.append(episode)

        self.time_step += 1
        if len(self.replay_memory) > type(self).REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > type(self).BATCH_SIZE:
            res = self.train()
            return res
        else:
            return None

    def minibatches(self):
        batch = random.sample(self.replay_memory, type(self).BATCH_SIZE)
        # state t
        state = self._np_shaping(np.array([item['obs'] for item in batch]), True)
        # action
        action = self._np_shaping(np.array([item['action'] for item in batch]), False)
        # reward
        reward = np.array([item['reward'] for item in batch])
        # state t+1
        state_2 = self._np_shaping(np.array([item['obs2'] for item in batch]), True)
        # doneA
        done = np.array([item['done'] for item in batch])

        i_reward = np.array([item['i_reward'] for item in batch])

        return state, action, reward, state_2, done , i_reward
    
    def get_in_reward(self, state):
        i_state = [0,0,0]
        _state = np.array(state)
        state_ = self._np_shaping(_state, True)
        i_state[0] = state_[0][0]
        i_state[1] = state_[0][1]
        i_state[2] = state_[0][2]
        i_state = np.array(i_state)
        i_state = self._np_shaping(i_state,True)
        a = self.target_net.target_net(i_state).astype(float)
        b = self.predict_net.predict_net(i_state).astype(float)
        in_reward = math.pow(math.fabs(a - b), 2)
        return in_reward

    def train(self):
        # sample a random minibatch of N transitions from R
        t_state, action, reward, t_state_2, done, i_reward = self.minibatches()

        actual_batch_size = len(t_state)

        target_action = self.actor_net.evaluate_target_actor(t_state)

        # Q'(s_i+1,a_i+1)
        q_t = self.critic_net.evaluate_target_critic(t_state_2, target_action)

        y = []
        for i in range(0, actual_batch_size):

            if done[i]:
                y.append(reward[i])
            else:
                y.append(reward[i] + type(self).GAMMA * q_t[i][0])  # q_t+1 instead of q_t

        y = np.reshape(np.array(y), [len(y), 1])

        ir = []
        for i in range(0, actual_batch_size):

            ir.append(i_reward[i])


        ir = np.reshape(np.array(ir), [len(ir), 1])

        # Update critic by minimizing the loss
        self.critic_net.train_critic(t_state, action, y)

        # Update actor proportional to the gradients:
        action_for_delQ = self.actor_net.evaluate_actor(t_state)  # dont need wolp action

        if self.is_grad_inverter:
            del_Q_a = self.critic_net.compute_delQ_a(t_state, action_for_delQ)  # /BATCH_SIZE
            del_Q_a = self.grad_inv.invert(del_Q_a, action_for_delQ)
        else:
            del_Q_a = self.critic_net.compute_delQ_a(t_state, action_for_delQ)[0]  # /BATCH_SIZE

        pre_q = np.array(self.predict_net.predict_gradient_input)
        pre_q_a = pre_q

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(t_state, del_Q_a)
        self.predict_net.train_predict(t_state, ir)

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
