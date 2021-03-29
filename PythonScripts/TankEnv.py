import numpy as np
import gym
from gym import spaces
import socket
import json
import time
from Agent import *
from stable_baselines3 import PPO
import random
from numpy.random import choice
import math

def choice_with_normalization(elements, weights):
    if sum(weights) == 0:
        return choice(elements, p=[1 for x in weights])
    return choice(elements, p=[x/sum(weights) for x in weights])

# return 1. if x is within D of 0, else return 1/(x/D)^2
def foo(x, D):
    if x == 0: return 1.
    x = 1. / math.pow(x/D, 2)
    return x if x <= 1. else 1.
    
def elo_based_choice(agents, center_elo, D):
    probs = [foo(e.elo-center_elo, D) for e in agents]
    return choice_with_normalization(agents, probs)

class TankEnv(gym.Env):
    metadata = {'render.modes': None}
    
    def __init__(self, num_agents=2, 
                        game_ip="127.0.0.1", 
                        game_port=50000, 
                        num_connection_attempts=60, 
                        sock_timeout=60., 
                        agent=0,
                        opp_buffer_size=1,
                        random_opp_sel=True,
                        center_elo=1000,
                        D=35.):
        super(TankEnv, self).__init__()
        
        self.opponent = None
        self.opp_buf_size = abs(opp_buffer_size)
        self.opponent_buf = []
        self.opp_num = 0
        self.opp_idx = 0
        self.random_opp_sel = random_opp_sel
        self.center_elo = center_elo
        self.D = D
        
        #self.action_trace = []
        #self.state_trace = []
        
        self.game_ip = game_ip
        self.game_port = game_port
        self.num_connection_attempts = num_connection_attempts
        self.sock_timeout = sock_timeout
        
        self.num_agents = num_agents
        self.action_space = spaces.Box(low=-1., high=1., shape=(num_agents, 5), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(num_agents, 26), dtype=np.float32)
        
        # Establish connection with Unity environment
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(num_connection_attempts):
            try:
                self.sock.connect((game_ip, game_port))
                break
            except ConnectionRefusedError:
                print("Could not connect to IP", game_ip, "on port", game_port, "...sleeping for one second")
                time.sleep(1)
        if i >= num_connection_attempts-1:
            print("Problem connecting to game on IP", game_ip, "on port", game_port)
            raise ConnectionError
        print("Connected to IP", game_ip, "on port", game_port)
        self.sock.settimeout(sock_timeout)
        
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
            print("Deep Agent opponent play mode indicated, make sure to explicitly load opponent policy.")
        else:
            self.opponent = Agent("stationary_agent", ob, (self.action_space.shape[1],))
        self.agent = agent
        
    def send(self, message):
        data = json.dumps(message)
        try:
            self.sock.sendall(bytes(data, encoding="utf-8"))
        except socket.timeout:
            print("Timeout while sending data, quitting now.")
            quit()
        #print("Sent:", data)
        
    def receive(self):
        try:
            received = self.sock.recv(1024)
        except socket.timeout:
            print("Timeout while receiving data, quitting now.")
            quit()
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
                    
                if self.agent == -1:
                    if self.random_opp_sel:
                        self.opponent = elo_based_choice(self.opponent_buf, self.center_elo, self.D)
                        print("I choose", self.opponent.name, "to play against with ELO", self.opponent.elo)
                    else:
                        self.opponent = self.opponent_buf[self.opp_idx]
                        self.opp_idx = (self.opp_idx + 1) % self.opp_num
                        #print("Next opponent to play against is", self.opponent.name)
                
                return self.state # TODO: Change this to handle multiple agents
            except json.decoder.JSONDecodeError:
                print("'Double JSON' error detected during reset, trying reset again...'")
                continue
            except ConnectionResetError:
                print("ConnectionResetError detected during reset, attempting to reconnect")
                # Establish connection with Unity environment
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                for i in range(self.num_connection_attempts):
                    try:
                        self.sock.connect((self.game_ip, self.game_port))
                        break
                    except ConnectionRefusedError:
                        print("Could not connect to IP", self.game_ip, "on port", self.game_port, "...sleeping for one second")
                        time.sleep(1)
                if i >= self.num_connection_attempts-1:
                    print("Problem connecting to game on IP", self.game_ip, "on port", self.game_port)
                    raise ConnectionError
                print("Connected to IP", self.game_ip, "on port", self.game_port)
                self.sock.settimeout(self.sock_timeout)
                print("Connection reestablished")
        
    def step(self, action):
        #self.action_trace.append(action)
        #self.state_trace.append(self.state)
        # Format actions into message for Unity
        #np.nan_to_num(action, copy=False)
        message = {}
        #for i,a in enumerate(action, start=1):
            #message[i] = a.tolist()
        message[1] = action.tolist()
        if self.agent != -1:
            message[2] = self.opponent.get_action(self.state).tolist()
        else:
            opp_state = np.concatenate([self.state[26:], self.state[:26]])
            old_action, _ = self.opponent.predict(opp_state)
            message[2] = old_action.tolist()
            
        #self.action_trace = self.action_trace[-5:]
        #self.state_trace = self.state_trace[-5:]
        
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
        
    def load_opp_policy(self, opp_name, elo=1000):
        print("Loading opponent policy named", opp_name, "with elo", elo)
        self.opponent = PPO.load(opp_name)
        self.opponent.name = opp_name.split('/')[-1]
        self.opponent.elo = elo
        self.opp_num += 1
        self.opponent_buf.append(self.opponent)
        self.opponent_buf = self.opponent_buf[-self.opp_buf_size:]