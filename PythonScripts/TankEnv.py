import numpy as np
import gym
from gym import spaces
import socket
import json
import time

class TankEnv(gym.Env):
    metadata = {'render.modes': None}
    
    def __init__(self, num_agents=2, game_ip="127.0.0.1", game_port=50000, num_connection_attempts=60):
        super(TankEnv, self).__init__()
        
        self.num_agents = num_agents
        self.action_space = spaces.Box(low=-1., high=1., shape=(num_agents, 5), dtype=np.float32)
        self.observation_space = spaces.Box(low=-8., high=8., shape=(num_agents, 26), dtype=np.float32)
        
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
        
    def send(self, message):
        data = json.dumps(message)
        self.sock.sendall(bytes(data, encoding="utf-8"))
        print("Sent:", data)
        
    def receive(self):
        received = self.sock.recv(1024)
        received = received.decode("utf-8")
        print("Received:", received)
        return json.loads(received)
        
    def reset(self):
        # Send start message
        self.send( {"start":True} )
        # Receive acknowledgement
        received = self.receive()
        if not "starting" in received:
            raise Exception("Something wrong with starting game")
        # Receive first state
        received = self.receive()
        state = np.array(received["state"])
        return state.reshape((self.num_agents, 26))
        
        
    def step(self, action):
        # Format actions into message for Unity
        message = {}
        for i,a in enumerate(action, start=1):
            message[i] = a.tolist()
        # Step Unity environment
        self.send(message)
        received = self.receive()
        
        state = np.array(received["state"])
        state = state.reshape((self.num_agents, 26))
        
        done = bool("done" in received)
        
        reward = np.zeros(self.num_agents)
        if "winner" in received:
            winner = int(received["winner"])
            if winner != -1:
                for i in range(self.num_agents):
                    reward[i] = 1 if winner == i else -1
                
        info = {}
        
        return state, reward, done, info
        
    def render(self, mode='console'):
        raise NotImplementedError()
        
    def close(self):
        self.send( {"end":True} )
        received = self.receive()
        if not "ending" in received:
            print("Something wrong with ending game")
        self.sock.close()