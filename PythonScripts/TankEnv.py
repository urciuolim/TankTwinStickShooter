import numpy as np
import gym
from gym import spaces
import socket
import json
import time
from Agent import *
from stable_baselines3 import SAC
import random

class TankEnv(gym.Env):
    metadata = {'render.modes': None}
    
    def __init__(self, num_agents=2, game_ip="127.0.0.1", game_port=50000, num_connection_attempts=60, agent=0, random_side=False):
        super(TankEnv, self).__init__()
        
        self.opponent = None
        
        self.num_agents = num_agents
        self.action_space = spaces.Box(low=-1., high=1., shape=(num_agents, 5), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(num_agents, 26), dtype=np.float32)
        self.rs = random_side
        self.p1 = 1
        self.p2 = 2
        
        # Establish connection with Unity environment
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(num_connection_attempts):
            try:
                self.sock.connect((game_ip, game_port))
                break
            except ConnectionRefusedError:
                print("Could not connect to IP", game_ip, "on port", game_port, "...sleeping for one second")
                time.sleep(1)
        print("Connected to IP", game_ip, "on port", game_port)
        
        ob = (num_agents*26,)
        ac = np.array([[-1., -1., -1., -1., 0], [1., 1., 1., 1., 1.]])       
        
        if agent > 0:
            if agent == 1:
                self.opponent = RandomAgent("random_agent", ob, ac)
            elif agent == 2:
                self.opponent = LineAgent("line_agent", ob, ac)
            elif agent == 3:
                self.opponent = BoxAgent("box_agent", ob, ac)
        elif agent == -1:
            print("Self play mode indicated, make sure to explicitly load old policy as opponent")
        else:
            self.opponent = Agent("stationary_agent", ob, (self.action_space.shape[1],))
        self.agent = agent
        
    def send(self, message):
        data = json.dumps(message)
        self.sock.sendall(bytes(data, encoding="utf-8"))
        #print("Sent:", data)
        
    def receive(self):
        received = self.sock.recv(1024)
        received = received.decode("utf-8")
        #print("Received:", received)
        return json.loads(received)
        
    def reset(self):
        while True:
            try:
                # Send restart message
                self.send( {"restart":True} )
                # Receive acknowledgement
                received = self.receive()
                # Send start message
                self.send( {"start":True} )
                # Receive acknowledgement
                received = self.receive()
                if not "starting" in received:
                    raise Exception("Something wrong with starting game")
                # Receive first state
                received = self.receive()
                self.state = np.array(received["state"])
                #return state.reshape((self.num_agents, 26))
                
                if self.opponent.name == "box_agent":
                    self.opponent.reset()
                    
                if self.rs:
                    self.p1 = random.randint(1,2)
                    self.p2 = 2 if self.p1 == 1 else 1
                    
                if self.p1 == 2:
                    self.state = np.concatenate([self.state[26:], self.state[:26]])
                
                return self.state # TODO: Change this to handle multiple agents
            except json.decoder.JSONDecodeError:
                print("'Double JSON' error detected during reset, trying reset again...'")
                continue
        
    def step(self, action):
        # Format actions into message for Unity
        message = {}
        #for i,a in enumerate(action, start=1):
            #message[i] = a.tolist()
        message[self.p1] = action.tolist()
        if self.agent != -1:
            message[self.p2] = self.opponent.get_action(self.state).tolist()
        else:
            opp_state = np.concatenate([self.state[26:], self.state[:26]])
            old_action, _ = self.opponent.predict(opp_state)
            message[self.p2] = old_action.tolist()
        
        # Step Unity environment
        self.send(message)
        received = self.receive()
        
        self.state = np.array(received["state"])
        #state = state.reshape((self.num_agents, 26))
        
        done = bool("done" in received)
        
        reward = 0#np.zeros(self.num_agents)
        if "winner" in received:
            winner = int(received["winner"])
            if winner != -1:
                reward = 1 if winner == 0 else -1
            '''
            if winner = -1:
                for i in range(self.num_agents):
                    reward[i] = 1 if winner == i else -1
            '''
                
        info = {}
        
        if self.p1 == 2:
            self.state = np.concatenate([self.state[26:], self.state[:26]])
        
        return self.state, reward, done, info
        
    def render(self, mode='console'):
        raise NotImplementedError()
        
    def close(self):
        # Send restart message
        self.send( {"restart":True} )
        # Receive acknowledgement
        received = self.receive()
        # Send end message
        self.send( {"end":True} )
        received = self.receive()
        if not "ending" in received:
            print("Something wrong with ending game")
        self.sock.close()
        
    def load_old_policy(self, oldname):
        print("Loading old policy named", oldname, "for selfplay")
        del self.opponent
        self.opponent = SAC.load(oldname)
        self.opponent.name = "selfplay_old_policy"