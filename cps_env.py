
import sys,os
sys.path.insert(0,os.getcwd())
import numpy as np
import gym


'''
    This file Provides an interface between the CPS system and the algorithm. You setup the CPS you 
    want to test.
    BTW: The simulator of SWAT can be acquired by getting in touch with .
'''



class cps():
    def __init__(self):             #observation_space and action_space should be set according to different CPS
        self.observation_space = gym.spaces.Box(low=-500.0, high=1500.0, shape=(1, 5), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 26), dtype=np.float32)

    def reset(self):                 # initialize the state with different CPS
        observation = self.Reset()
        return observation

    def step(self,action,observation,flag,t):       #apply action to CPS, get next state, reward and signal of end
        reward = 0
        done = False
        next_observation = [0 in range(np.size(observation))]

        return next_observation,reward,done



