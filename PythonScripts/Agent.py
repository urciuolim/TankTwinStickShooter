from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, name, state_dim, action_dim):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    @abstractmethod
    def get_action(self, state):
        pass