import numpy as np
import pyflann
from gym.spaces import Box
from src.ddpg import agent
from src import action_space


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions, k_ratio,l):

        super().__init__(env,l)
        self.experiment = 'swat'
        if self.continious_action_space:
            self.action_space = action_space.Space(self.low, self.high, max_actions)
            max_actions = self.action_space.get_number_of_actions()
        else:
            max_actions = int(env.action_space.n)
            self.action_space = action_space.Discrete_space(max_actions)

        self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))
        print(self.k_nearest_neighbors)

    def get_name(self):
        return 'Wolp_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def act(self, state):
        proto_action = super().act(state)
        return self.wolp_action(state, proto_action)

    def wolp_action(self, state, proto_action):
        actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)[0]
        self.data_fetch.set_ndn_action(actions[0].tolist())
        states = np.tile(state, [len(actions), 1])
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        max_index = np.argmax(actions_evaluation)
        action = actions[max_index]
        return action
