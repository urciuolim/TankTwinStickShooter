import numpy as np
from numpy.random import choice
import gym
from gym import spaces
from stable_baselines3 import PPO
import socket
import json
import time
import random
import math
import subprocess

def choice_with_normalization(elements, weights):
    if sum(weights) == 0:
        return choice(elements, p=[1 for x in weights])
    return choice(elements, p=[x/sum(weights) for x in weights])

# return 1. if x is within D of 0, else return 1/(x/D)^2
def weight_func(x, D):
    if x == 0: return 1.
    x = 1. / math.pow(x/D, 2)
    return x if x <= 1. else 1.
    
def elo_based_choice(opponent_elos, center_elo, D):
    weights = [weight_func(elo-center_elo, D) for elo in opponent_elos]
    return choice_with_normalization([i for i in range(len(opponent_elos))], weights)

class TankEnv(gym.Env):
    metadata = {'render.modes': None}

    def __init__(self, game_path, 
                        opp_fp_and_elo=[],
                        game_log_path="gamelog.txt",
                        game_ip="127.0.0.1", 
                        game_port=50000,
                        my_port=None,
                        num_connection_attempts=60, 
                        sock_timeout=60., 
                        elo_match=True,
                        center_elo=1000,
                        D=35.,
                        survivor=False,
                        max_steps=300
                ):
        super(TankEnv, self).__init__()
        
        #TODO: save opponent models and elos separately
        # opp_fp_and_elo = [] allows starting of env without loading opponents
        if len(opp_fp_and_elo) > 0:
            print("Env", game_port, "is starting to load opponents...", flush=True)
            self.opponents = []
            for (opp_fp, elo) in opp_fp_and_elo:
                self.opponents.append((PPO.load(opp_fp), elo, opp_fp))
                print("Env", game_port, "loaded", opp_fp, "with elo", elo, flush=True)
            print("Env", game_port, "has finished loading", len(self.opponents), "opponents", flush=True)

        self.curr_opp = 0
        self.elo_match=elo_match
        self.center_elo = center_elo
        self.D = D
        
        self.survivor = survivor
        self.step_counter = 0
        self.max_steps = max_steps
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(52,), dtype=np.float32)
        
        self.my_port = my_port if my_port else game_port + 1
        self.num_connection_attempts = num_connection_attempts
        self.game_port = game_port
        self.game_ip = game_ip
        self.sock_timeout = sock_timeout
        self.game_path = game_path
        # Open text file for game logging output
        self.game_log = open(game_log_path, 'w')
        
        # Establish connection with Unity environment
        self.sock = self.connect_to_unity()
        
    def fix_connection(self):
        self.sock.close()
        if self.game_path:
            self.game_p.terminate()
        self.sock = self.connect_to_unity()
        
    def send(self, message):
        data = json.dumps(message)
        try:
            self.sock.sendall(bytes(data, encoding="utf-8"))
        except socket.timeout:
            print("Timeout while sending data, quitting now.", flush=True)
            raise ConnectionError
        
    def receive(self):
        try:
            received = self.sock.recv(1024)
        except socket.timeout:
            print("Timeout while receiving data, quitting now.", flush=True)
            raise ConnectionError
        received = received.decode("utf-8")
        return json.loads(received)
        
    def connect_to_unity(self):
        if self.game_path:
            # Start game executable
            game_cmd_list = [self.game_path, str(self.game_port)]
            self.game_p = subprocess.Popen(game_cmd_list, stdout=self.game_log, stderr=self.game_log)
        # Init socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow port to be used by socket to be reused after disconnection
        #   (OS will allow resuse of port even if it is in TIME_WAIT status)
        #   (DOES NOT WORK ON WINDOWS)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Attempt to bind my end of socket connection to my_port
        os_error = None
        for i in range(self.num_connection_attempts):
            try:
                sock.bind(("", self.my_port))
                break
            except OSError as os_error:
                print("OSError during binding, trying again...", flush=True)
                time.sleep(1)
            except BrokenPipeError:
                print("BrokenPipeError during binding, trying again...", flush=True)
                time.sleep(1)
        if i >= self.num_connection_attempts-1:
            print("Problem binding on port", self.my_port, ", trying random ports", flush=True)
            for k in range(self.num_connection_attempts):
                try:
                    sock.bind(("", random.randint(33000, 60000)))
                    break
                except OSError as os_error:
                    print("OSError during binding, trying again...", flush=True)
                    time.sleep(1)
                except BrokenPipeError:
                    print("BrokenPipeError during binding, trying again...", flush=True)
                    time.sleep(1)
            if k >= self.num_connection_attempts-1:
                print("Still problem binding on ports, I give up now :(", flush=True)
                raise os_error if os_error else OSError
            
        # Attempt to connect to Unity game instance
        for j in range(self.num_connection_attempts):
            try:
                sock.connect((self.game_ip, self.game_port))
                break
            except ConnectionRefusedError:
                print("Could not connect to IP", self.game_ip, "on port", self.game_port, "...sleeping for one second", flush=True)
                time.sleep(1)
            except OSError:
                print("OSError during connect, trying again...", flush=True)
                time.sleep(1)
            except BrokenPipeError:
                print("BrokenPipeError during connect, trying again...", flush=True)
                time.sleep(1)
        if j >= self.num_connection_attempts-1:
            print("Problem connecting to game on IP", self.game_ip, "on port", self.game_port, flush=True)
            raise ConnectionError
        
        print("Connected to IP", self.game_ip, "on port", self.game_port, "with my_port = ", self.my_port, flush=True)
        # Set socket timeout
        sock.settimeout(self.sock_timeout)
        return sock
        
    def reset(self):
        ELO=1
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
                self.step_counter = 0
                    
                if self.elo_match:
                    self.curr_opp = elo_based_choice([opp[ELO] for opp in self.opponents], self.center_elo, self.D)
                
                return self.state
            except json.decoder.JSONDecodeError:
                print("'Double JSON' error detected during reset, trying reset again...'", flush=True)
                continue
            except ConnectionResetError:
                print("ConnectionResetError detected during reset, attempting to reconnect", flush=True)
                # Retry connection with Unity environment
                self.sock = self.connect_to_unity()
                print("Connection reestablished", flush=True)
            except ConnectionError:
                print("ConnectionError during reset, attempting to reconnect...", flush=True)
                self.fix_connection()
                
    def next_opp(self):
        FP=2
        self.curr_opp += 1
        print("Next opponent is", self.opponents[self.curr_opp][FP], flush=True)
        
    def step(self, action):
        POLICY=0
        PLAYER_1=0
        # Format actions into message for Unity
        message = {}
        # Inputted action
        message[1] = action.tolist()
        # Opponent action, reversing the order of the state to the perspective of the opponent
        opp_state = np.concatenate([self.state[26:], self.state[:26]])
        opp_action, _ = self.opponents[self.curr_opp][POLICY].predict(opp_state)
        message[2] = opp_action.tolist()
        
        # Step Unity game environment
        try:
            self.send(message)
            received = self.receive()
        except ConnectionError:
            print("ConnectionError during step, counting this game as ending with no winner and then reconnecting...", flush=True)
            self.fix_connection()
            return self.state, 0, True, {"lost_connection":True}
        
        self.state = np.array(received["state"])
        self.step_counter += 1
        
        done = bool("done" in received)
        info = {}
        
        reward = 0
        if done or "winner" in received:
            done = True
            if "winner" in received:
                winner = int(received["winner"])
                if winner != -1:
                    reward = 1 if winner == PLAYER_1 else -1
                info["winner"] = winner
                    
        if self.survivor and done:
            reward = self.step_counter / self.max_steps
        
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
            print("Something wrong with ending game", flush=True)
        self.sock.close()
        if self.game_path:
            self.game_p.wait()
        self.game_log.close()
        print("Game", self.game_port, "closed port", flush=True)