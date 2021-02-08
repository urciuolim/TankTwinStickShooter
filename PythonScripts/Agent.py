from abc import ABC, abstractmethod
import numpy as np
import random

class Agent(ABC):
    def __init__(self, name, state_dim, action_dim):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def get_action(self, state):
        return np.zeros(self.action_dim)
        

class RandomAgent(Agent):
    def __init__(self, name, state_dim, action_lim):
        super().__init__(name, state_dim, action_lim[0,:].shape)
        self.action_lim = action_lim
        
    def get_action(self, state):
        action_mins = self.action_lim[0,:]
        action_maxs = self.action_lim[1,:]
        action = np.random.rand(*self.action_dim)
        action = (action_maxs - action_mins) * action + action_mins
        return action
        
class LineAgent(RandomAgent):
    def __init__(self, name, state_dim, action_lim, right=True):
        super().__init__(name, state_dim, action_lim)
        self.move = random.choice([-1, 1])
        self.right = right
        
    def get_action(self, state):
        action = super().get_action(state)
        # No horizontal movement
        action[0] = 0
        # Vertical movement of hard-coded width of arena
        if state[27] > 3:
            self.move = -1
        elif state[27] < -3:
            self.move = 1
        elif np.random.choice([True, False], 1, p=[.1, .9])[0]:
            self.move *= -1
        action[1] = self.move
        # Horizontal aiming direction
        action[2] = -1 if self.right else 1
        # Always be shootin
        action[4] = 1
        return action
        
class BoxAgent(RandomAgent):
    def __init__(self, name, state_dim, action_lim, right=True):
        super().__init__(name, state_dim, action_lim)
        self.clockwise = [[0, -1], [-1, 0], [0, 1], [1, 0]]
        self.counterclockwise = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.reset()
        
    def reset(self):
        self.ptr = 0
        self.is_clockwise = random.choice([True, False])
        if self.is_clockwise:
            self.move_plan = self.clockwise
        else:
            self.move_plan = self.counterclockwise
        self.in_corner = False
        
    def get_action(self, state):
        action = super().get_action(state)
        
        horz = state[26]
        vert = state[27]
        
        # Determine if at corners, if so change to next step in move plan
        if (self.ptr % 2 == 0 and abs(vert) >= 3) or (self.ptr % 2 == 1 and abs(horz) >= 7):
            if not self.in_corner:
                self.ptr += 1
                self.ptr %= len(self.move_plan)
                self.in_corner = True
        else:
            self.in_corner = False
        move = self.move_plan[self.ptr]
        move = np.array(move)
        action[:2] = move
        
        # Always be shootin
        action[4] = 1
        return action