from Agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self, name, state_dim, action_lim):
        super(self.__class__, self).__init__(name, state_dim, action_lim[0,:].shape)
        self.action_lim = action_lim
        
    def get_action(self, state):
        action_mins = self.action_lim[0,:]
        action_maxs = self.action_lim[1,:]
        action = np.random.rand(*self.action_dim)
        action = (action_maxs - action_mins) * action + action_mins
        return action