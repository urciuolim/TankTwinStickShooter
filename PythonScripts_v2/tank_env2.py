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
from subprocess import Popen, PIPE, STDOUT

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
                        elo_match=True,
                        center_elo=1000,
                        D=35.,
                        survivor=False,
                        max_steps=300
                ):
        super(TankEnv, self).__init__()
        
        game_port = -1
        self.game_port = game_port
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
        
        self.game_path = game_path
        # Open text file for game logging output
        self.game_log = open(game_log_path, 'w')
        
        # Establish connection with Unity environment
        self.open_game()
        
    def open_game(self):
        if self.game_path:
            # Start game executable
            self.game_p = Popen(self.game_path, bufsize=1, stdin=PIPE, stdout=self.game_log, stderr=self.game_log, text=True)
            time.sleep(20)
            
    def send(self, message):
        data = json.dumps(message) + '\n'
        self.game_p.stdin.write(data)
        '''
        try:
            self.sock.sendall(bytes(data, encoding="utf-8"))
        except socket.timeout:
            print("Timeout while sending data, quitting now.", flush=True)
            raise ConnectionError
        '''
        
    def receive(self):
        while True:
            out = self.game_p.stdout.readline()
            if out[0:11] == "FOR_PYTHON|":
                return out[11:-1]
            else:
                self.game_log.write(out)
        '''
        try:
            received = self.sock.recv(1024)
        except socket.timeout:
            print("Timeout while receiving data, quitting now.", flush=True)
            raise ConnectionError
        received = received.decode("utf-8")
        return json.loads(received)
        '''
    
    def message_unity(self, message):
        message = str(message) + '\n'
        self.game_p.stdin.write(message)
        while True:
            out = self.game_p.stdout.readline()
            if out[0:11] == "FOR_PYTHON|":
                return out[11:-1]
            else:
                self.game_log.write(out)
        
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
        if self.game_path:
            self.game_p.terminate()
        self.game_log.close()