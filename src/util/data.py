import json
import os
from os.path import basename
import zipfile


def load(file_name,flag = 1):
    data = Data(flag)
    if zipfile.is_zipfile(file_name):
        print('Data: Unziping ', file_name, '...')
        with zipfile.ZipFile(file_name) as myzip:
            string = (myzip.read(myzip.namelist()[0]).decode("utf-8"))
            data.set_data(json.loads(string))
    else:
        print('Data: Loading ', file_name, '...')
        with open(file_name, 'r') as f:
            data.set_data(json.load(f))
    return data


class Data:

    PATH = 'results/obj/'
    AUTOSAVE_BATCH_SIZE = 1e5

    DATA_TEMPLATE = '''
    {
        "id":0,
        "time":0,
        "agent":{
          "name":"default_name",
          "max_actions":0,
          "k":0,
          "version":0
        },
        "experiment":{
          "name":"no_exp",
          "actions_low":null,
          "actions_high":null,
          "number_of_episodes":0
        },
        "simulation":{
          "episodes":[]
        }
    }
    '''

    EPISODE_TEMPLATE = '''
    {
        "id":0,
        "i_rewards":[],
        "e_rewards":[],
        "rewards":[],
        "action":[],
        "state":[],
        "ndn_actions":[],
        "actors_actions":[],
        "steps":0,
        "ep_time":0
    }
    '''


    def __init__(self,flag):
        self.path = 'results/obj_{}/'.format(flag)            # It stores different failure in different files, but you need create a corresponding file first.
        self.data = json.loads(self.DATA_TEMPLATE)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id = 0
        self.temp_saves = 0
        self.data_added = 0

    def __increase_data_counter(self, n=1):
        self.data_added += n

    def set_id(self, n):
        self.data['id'] = n

    def set_agent(self, name, max_actions, k, version):
        self.data['agent']['name'] = name
        self.data['agent']['max_actions'] = max_actions
        self.data['agent']['k'] = k
        self.data['agent']['version'] = version

    def set_experiment(self, name, low, high, eps):
        self.data['experiment']['name'] = name
        self.data['experiment']['actions_low'] = low
        self.data['experiment']['actions_high'] = high
        self.data['experiment']['number_of_episodes'] = eps


    def set_actors_action(self, action):
        self.episode['actors_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_ndn_action(self, action):
        self.episode['ndn_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_reward(self, reward):
        self.episode['rewards'].append(reward)
        self.__increase_data_counter()

    def set_i_reward(self, reward):
        self.episode['i_rewards'].append(reward)
        self.__increase_data_counter()

    def set_e_reward(self, reward):
        self.episode['e_rewards'].append(reward)
        self.__increase_data_counter()

    def set__action(self, action):
        self.episode['action'].append(action)
        self.__increase_data_counter(len(action))

    def set__state(self, state):
        self.episode['state'].append(state)
        self.__increase_data_counter(len(state))

    def end_of_episode(self):
        self.data['simulation']['episodes'].append(self.episode)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id += 1
        self.episode['id'] = self.episode_id

    def finish_and_store_episode(self,exp,l_r,flag,i):
        self.end_of_episode()
        if self.data_added > self.AUTOSAVE_BATCH_SIZE:
            self.temp_save(exp = exp,l_r = l_r,flag = flag,i=i)

    def get_file_name(self,i = 0,exp = 0.7,l = 0.0001):
        return 'data_{}_{}_{}{}k{}#{}_{}_{}_{}'.format(self.get_episodes(),
                                              self.get_agent_name(),
                                              self.get_experiment()[:3],
                                              self.data['agent']['max_actions'],
                                              self.data['agent']['k'],
                                              self.get_id(),
                                                    exp,
                                                    l,
                                                       i)

    def get_episodes(self):
        return self.data['experiment']['number_of_episodes']

    def get_agent_name(self):
        return '{}{}'.format(self.data['agent']['name'][:4],
                             self.data['agent']['version'])

    def get_id(self):
        return self.data['id']

    def get_experiment(self):
        return self.data['experiment']['name']

    def print_data(self):
        print(json.dumps(self.data, indent=2, sort_keys=True))

    def print_stats(self):
        for key in self.data.keys():
            d = self.data[key]
            if key == 'simulation':
                print('episodes:', len(d['episodes']))
            else:
                print(json.dumps(d, indent=2, sort_keys=True))

    def merge(self, data_in):
        if type(data_in) is Data:
            data = data_in.data
        else:
            data = data_in

        for ep in data['simulation']['episodes']:
            self.episode = ep
            self.end_of_episode()

    def set_data(self, data):
        self.data = data

    def save(self, j, exp ,l_r,flag, path='', final_save=True):
        if final_save and self.temp_saves > 0:
            if self.data_added > 0:
                self.end_of_episode()
                self.temp_save(exp = exp,l_r = l_r,flag = flag,i=j)
            print('Data: Merging all temporary files')
            for i in range(self.temp_saves):
                file_name = '{}temp/{}{}.json'.format(self.path,
                                                      i,
                                                      self.get_file_name(exp = exp,l = l_r,i = j))
                temp_data = load(file_name,flag)
                self.merge(temp_data)
                os.remove(file_name)

        final_file_name = self.path + path + self.get_file_name(exp = exp,l = l_r,i = j) + '.json'
        if final_save:
            print('Data: Ziping', final_file_name)
            with zipfile.ZipFile(final_file_name + '.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
                myzip.writestr(basename(final_file_name), json.dumps(
                    self.data, indent=2, sort_keys=True))
        else:
            with open(final_file_name, 'w') as f:
                print('Data: Saving', final_file_name)
                json.dump(self.data, f)

    def temp_save(self,exp, l_r,flag,i):
        if self.data_added == 0:
            return
        self.save(path='temp/' + str(self.temp_saves), final_save=False,exp = exp,l_r = l_r,flag = flag,j=i)
        self.temp_saves += 1
        self.data['simulation']['episodes'] = []
        self.data_added = 0

    def set_time(self, n):
        self.data['time'] = n

    def set_step(self, n):
        self.episode['steps'] = n

    def set_ep_time(self, n):
        self.episode['ep_time'] = n




# if __name__ == '__main__':
#
#     import numpy as np
#     import random
#
#     # d = load('results/obj/saved/data_10001_Wolp3_InvertedPendulum-v1#0.json.zip')
#     # # d = load('results/obj/saved/data_10000_agent_name4_exp_name#0.json.zip')
#     # print(d.get_file_name())
#     # d = load('results/obj/data_10000_agent_name4_exp_name#0.json.zip')
#     d = Data()
#     d.set_agent('agent_name', 1000, 10, 4)
#     d.set_experiment('exp_name', [-2, -3], [3, 2], 10000)
#
#     # d.print_data()
#     #
#     for i in range(10):
#         d.set_state([i, i, i, i])
#         d.set_action([i, i])
#         d.set_actors_action([i, i])
#         d.set_ndn_action([i, i])
#         d.set_reward(i)
#         if i % 3 == 0:
#             d.finish_and_store_episode()
#             d.temp_save()
#             # exit()
#
#     # for i in range(30, 400):
#     #     d.set_state([i, i, i, i])
#     #     d.set_action([i, i])
#     #     d.set_actors_action([i, i])
#     #     d.set_ndn_action([i, i])
#     #     d.set_reward(random.randint(0, 10))
#     #     if i % 2 == 0:
#     #         d.finish_and_store_episode()
#     #         d.temp_save()
#     # #
#     d.temp_save()
#     d.temp_save()
#     d.save()
#    d.print_data()
