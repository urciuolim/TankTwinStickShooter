import numpy as np
import gym
from gym import spaces

class IndvTankEnv(gym.Env):
    metadata = {'render.modes': None}
    
    def __init__(self, env):
        super(IndvTankEnv, self).__init__()
        
        self.env = env
        self.action_space = spaces.Box(low=-1., high=1., shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(self.env.num_agents*26,), dtype=np.float32)
        
    def reset(self):
        return self.env.reset()
        
        
    def step(self, action):
        return self.env.step(action)
        
    def render(self, mode='console'):
        raise NotImplementedError()
        
    def close(self):
        self.env.close()
        
    def load_opp_policy(self, oldname):
        self.env.load_opp_policy(oldname)