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
import sys
import os

def choice_with_normalization(elements, weights):
    if sum(weights) == 0:
        return choice(elements, p=[1/len(elements) for _ in elements])
    return choice(elements, p=[x/sum(weights) for x in weights])

# return 1. if x is within D of 0, else return 1/(x/D)^2
def weight_func(x, D):
    if x == 0: return 1.
    x = 1. / math.pow(x/D, 2)
    return x if x <= 1. else 1.
    
def elo_based_choice(opponent_elos, center_elo, D):
    weights = [weight_func(elo-center_elo, D) for elo in opponent_elos]
    return choice_with_normalization([i for i in range(len(opponent_elos))], weights)
    
def flip_state(state):
    R,G,B = (0,1,2)
    new_state = np.zeros(state.shape, dtype=state.dtype)
    new_state[:,:,R] = state[:,:,B].copy()
    new_state[:,:,G] = state[:,:,G].copy()
    new_state[:,:,B] = state[:,:,R].copy()
    return new_state

class TankEnv(gym.Env):
    metadata = {'render.modes': None}

    def __init__(self, game_path, 
                        opp_fp_and_elo=[],
                        game_log_path="gamelog.txt",
                        game_ip="127.0.0.1", 
                        game_port=50000,
                        my_port=None,
                        num_connection_attempts=60, 
                        sock_timeout=10., 
                        elo_match=True,
                        center_elo=1000,
                        D=35.,
                        survivor=False,
                        max_steps=300,
                        stdout_path=None,
                        verbose=False,
                        level_path=False,
                        image_based=False,
                        time_reward=0.,
                        rand_opp=False,
                        p=3,
                        opp_p=None
                ):
        super(TankEnv, self).__init__()
        
        os.environ["OMP_NUM_THREADS"] = "1"
        
        self.verbose=verbose
        self.stdout_path = None
        if stdout_path and self.verbose:
            self.stdout_path = stdout_path
            sys.stdout = open(stdout_path, 'a')
        
        self.rand_opp = rand_opp
        #TODO: save opponent models and elos separately
        # opp_fp_and_elo = [] allows starting of env without loading opponents
        self.opponents = []
        if len(opp_fp_and_elo) > 0 and not rand_opp:
            if self.verbose:
                print("Env", game_port, "is starting to load opponents...", flush=True)
            for (opp_fp, elo) in opp_fp_and_elo:
                self.opponents.append((PPO.load(opp_fp), elo, opp_fp))
                if self.verbose:
                    print("Env", game_port, "loaded", opp_fp, "with elo", elo, flush=True)
            if self.verbose:
                print("Env", game_port, "has finished loading", len(self.opponents), "opponents", flush=True)

        self.curr_opp = 0
        self.elo_match=elo_match
        self.center_elo = center_elo
        self.D = D
        self.p = p
        self.opp_p = opp_p if opp_p != None else self.p
        
        self.survivor = survivor
        self.step_counter = 0
        self.max_steps = max_steps
        
        self.image_based = image_based
        self.action_space = spaces.Box(low=-1., high=1., shape=(5,), dtype=np.float32)
        if not image_based:
            self.observation_space = spaces.Box(low=-100., high=10., shape=(52,), dtype=np.float32)
        else:
            if level_path:
                self.observation_space, self.state = self.load_level(level_path, p=self.p)
                if self.opp_p == self.p:
                    self.opp_state = flip_state(self.state)
                else:
                    _,self.opp_state = self.load_level(level_path, p=self.opp_p)
                    self.opp_state = flip_state(self.opp_state)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=(12*3,20*3,3), dtype=np.uint8)
        # Reward (or penalty) given at each time step
        self.time_reward = time_reward
        
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
        
        self.last_winner = -1
        
    def load_new_opp(self, idx, opp_fp, opp_elo):
        if idx < len(self.opponents):
            self.opponents[idx] = (PPO.load(opp_fp), opp_elo, opp_fp)
            self.curr_opp = idx
        else:
            self.opponents.append((PPO.load(opp_fp), opp_elo, opp_fp))
            self.curr_opp = len(self.opponents)-1
        
    def load_level(self, level_path, p=3):
        R,G,B=(0,1,2)
        with open(level_path, 'r') as level_file:
            level_json = json.load(level_file)
        dims = level_json["Walls"]["dims"]
        # p^2 = number of pixels to represent one grid square in game
        width = (dims["maxX"] - dims["minX"] + 1) * p
        height = (dims["maxY"] - dims["minY"] + 1) * p
        
        state = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(dims["minX"], dims["maxX"]+1):
            x_p = (x - dims["minX"]) * p
            for y in level_json["Walls"][str(x)]:
                y_p = (y - dims["minY"]) * p
                state[y_p:y_p+p, x_p:x_p+p, G] = 255
               
        self.dims = dims
            
        return spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8), state
        
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
            if self.verbose:
                print("Timeout while sending data, quitting now.", flush=True)
            raise ConnectionError
        
    def receive(self):
        try:
            received = self.sock.recv(1024)
        except socket.timeout:
            if self.verbose:
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
                if self.verbose:
                    print("OSError during binding, trying again...", flush=True)
                time.sleep(1)
            except BrokenPipeError:
                if self.verbose:
                    print("BrokenPipeError during binding, trying again...", flush=True)
                time.sleep(1)
        if i >= self.num_connection_attempts-1:
            if self.verbose:
                print("Problem binding on port", self.my_port, ", trying random ports", flush=True)
            for k in range(self.num_connection_attempts):
                try:
                    sock.bind(("", random.randint(33000, 60000)))
                    break
                except OSError as os_error:
                    if self.verbose:
                        print("OSError during binding, trying again...", flush=True)
                    time.sleep(1)
                except BrokenPipeError:
                    if self.verbose:
                        print("BrokenPipeError during binding, trying again...", flush=True)
                    time.sleep(1)
            if k >= self.num_connection_attempts-1:
                if self.verbose:
                    print("Still problem binding on ports, I give up now :(", flush=True)
                raise os_error if os_error else OSError
            
        # Attempt to connect to Unity game instance
        for j in range(self.num_connection_attempts):
            try:
                sock.connect((self.game_ip, self.game_port))
                break
            except ConnectionRefusedError:
                if self.verbose:
                    print("Could not connect to IP", self.game_ip, "on port", self.game_port, "...sleeping for one second", flush=True)
                time.sleep(1)
            except OSError as e:
                if self.verbose:
                    print("OSError during connect, trying again...", flush=True)
                    print(e)
                time.sleep(1)
            except BrokenPipeError:
                if self.verbose:
                    print("BrokenPipeError during connect, trying again...", flush=True)
                time.sleep(1)
        if j >= self.num_connection_attempts-1:
            if self.verbose:
                print("Problem connecting to game on IP", self.game_ip, "on port", self.game_port, flush=True)
            raise ConnectionError
        
        if self.verbose:
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
                self.raw_state = np.array(received["state"])
                if self.image_based:
                    self.state = self.draw_state(self.raw_state, self.state, self.p)
                else:
                    self.state = self.raw_state
                self.step_counter = 0
                    
                if self.elo_match and not self.rand_opp:
                    self.curr_opp = elo_based_choice([opp[ELO] for opp in self.opponents], self.center_elo, self.D)
                
                return self.state
            except json.decoder.JSONDecodeError:
                if self.verbose:
                    print("'Double JSON' error detected during reset, trying reset again...'", flush=True)
                continue
            except ConnectionResetError:
                if self.verbose:
                    print("ConnectionResetError detected during reset, attempting to reconnect", flush=True)
                # Retry connection with Unity environment
                self.sock = self.connect_to_unity()
                if self.verbose:
                    print("Connection reestablished", flush=True)
            except ConnectionError:
                if self.verbose:
                    print("ConnectionError during reset, attempting to reconnect...", flush=True)
                self.fix_connection()
                
    def next_opp(self):
        FP=2
        self.curr_opp += 1
        if self.verbose:
            print("Next opponent is", self.opponents[self.curr_opp][FP], flush=True)
            
    def kill_env(self):
        if self.game_path:
            self.game_p.kill()
            
    def draw_state(self, raw_state, last_state, p):
        R,G,B = (0,1,2)
        POS,VEC,AIM = (120,75,30)
        BUL_POS = 100
        state = np.zeros(last_state.shape, dtype=np.uint8)
        for y in range(last_state.shape[0]):
            for x in range(last_state.shape[1]):
                if last_state[y,x,G] >= 255:
                    state[y,x,G] = 255
        
        # Parse player 1 position/velocity/aiming direction
        p1_pos_x = np.clip(int((raw_state[0] - self.dims["minX"]) * p), 0, state.shape[1]-1)
        p1_pos_y = np.clip(int((raw_state[1] - self.dims["minY"]) * p), 0, state.shape[0]-1)
        p1_vec_x = np.clip(p1_pos_x + int(raw_state[2] * p), 0, state.shape[1]-1)
        p1_vec_y = np.clip(p1_pos_y + int(raw_state[3] * p), 0, state.shape[0]-1)
        p1_aim_x = np.clip(p1_pos_x + int(raw_state[4] * p), 0, state.shape[1]-1)
        p1_aim_y = np.clip(p1_pos_y + int(raw_state[5] * p), 0, state.shape[0]-1)

        # Put that info into red channel
        state[p1_pos_y, p1_pos_x, R] = min(state[p1_pos_y, p1_pos_x, R] + POS, 255)
        state[p1_vec_y, p1_vec_x, R] = min(state[p1_vec_y, p1_vec_x, R] + VEC, 255)
        state[p1_aim_y, p1_aim_x, R] = min(state[p1_aim_y, p1_aim_x, R] + AIM, 255)
        # Parse each of five possible player 1 bullets, again putting info into red channel
        for i in range(6, 26, 4):
            p1_bullet_pos_x = np.clip(int((raw_state[i] - self.dims["minX"]) * p), -100, state.shape[1]-1)
            p1_bullet_pos_y = np.clip(int((raw_state[i+1] - self.dims["minY"]) * p), -100, state.shape[0]-1)
            p1_bullet_vec_x = np.clip(p1_bullet_pos_x + int(raw_state[i+2] * p), -100, state.shape[1]-1)
            p1_bullet_vec_y = np.clip(p1_bullet_pos_y + int(raw_state[i+3] * p), -100, state.shape[0]-1)
            if p1_bullet_pos_x >= 0:
                state[p1_bullet_pos_y, p1_bullet_pos_x, R] = min(state[p1_bullet_pos_y, p1_bullet_pos_x, R] + BUL_POS, 255)
                state[p1_bullet_vec_y, p1_bullet_vec_x, R] = min(state[p1_bullet_vec_y, p1_bullet_vec_x, R] + VEC, 255)
            
        # Parse player 2 position/velocity/aiming direction
        p2_pos_x = np.clip(int((raw_state[26] - self.dims["minX"]) * p), 0, state.shape[1]-1)
        p2_pos_y = np.clip(int((raw_state[27] - self.dims["minY"]) * p), 0, state.shape[0]-1)
        p2_vec_x = np.clip(p2_pos_x + int(raw_state[28] * p), 0, state.shape[1]-1)
        p2_vec_y = np.clip(p2_pos_y + int(raw_state[29] * p), 0, state.shape[0]-1)
        p2_aim_x = np.clip(p2_pos_x + int(raw_state[30] * p), 0, state.shape[1]-1)
        p2_aim_y = np.clip(p2_pos_y + int(raw_state[31] * p), 0, state.shape[0]-1)
        # Put that info into blue channel
        state[p2_pos_y, p2_pos_x, B] = min(state[p2_pos_y, p2_pos_x, B] + POS, 255)
        state[p2_vec_y, p2_vec_x, B] = min(state[p2_vec_y, p2_vec_x, B] + VEC, 255)
        state[p2_aim_y, p2_aim_x, B] = min(state[p2_aim_y, p2_aim_x, B] + AIM, 255)
        # Parse each of five possible player 2 bullets, again putting info into red channel
        for i in range(32, 52, 4):
            p2_bullet_pos_x = np.clip(int((raw_state[i] - self.dims["minX"]) * p), -100, state.shape[1]-1)
            p2_bullet_pos_y = np.clip(int((raw_state[i+1] - self.dims["minY"]) * p), -100, state.shape[0]-1)
            p2_bullet_vec_x = np.clip(p2_bullet_pos_x + int(raw_state[i+2] * p), -100, state.shape[1]-1)
            p2_bullet_vec_y = np.clip(p2_bullet_pos_y + int(raw_state[i+3] * p), -100, state.shape[0]-1)
            if p2_bullet_pos_x >= 0:
                state[p2_bullet_pos_y, p2_bullet_pos_x, B] = min(state[p2_bullet_pos_y, p2_bullet_pos_x, B] + BUL_POS, 255)
                state[p2_bullet_vec_y, p2_bullet_vec_x, B] = min(state[p2_bullet_vec_y, p2_bullet_vec_x, B] + VEC, 255)
                
        return state
        
    def step(self, action):
        R,G,B = (0,1,2)
        POLICY=0
        PLAYER_1=0
        # Format actions into message for Unity
        message = {}
        # Inputted action
        message[1] = action.tolist()
        # Opponent action, reversing the order of the state to the perspective of the opponent
        if not self.image_based:
            self.opp_state = np.concatenate([self.state[26:], self.state[:26]])
        else:
            if self.opp_p == self.p:
                self.opp_state = flip_state(self.state)
            else:
                self.opp_state = flip_state(self.draw_state(self.raw_state, self.opp_state, self.opp_p))
        if self.rand_opp:
            opp_action = (np.random.rand(5) * 2) - 1
        else:
            opp_action, _ = self.opponents[self.curr_opp][POLICY].predict(self.opp_state)
        message[2] = opp_action.tolist()
        
        # Step Unity game environment
        try:
            self.send(message)
            received = self.receive()
        except ConnectionError:
            if self.verbose:
                print("ConnectionError during step, counting this game as ending with no winner and then reconnecting...", flush=True)
            self.fix_connection()
            return self.state, 0, True, {"lost_connection":True}
        
        self.raw_state = np.array(received["state"])
        if self.image_based:
            self.state = self.draw_state(self.raw_state, self.state, self.p)
        else:
            self.state = self.raw_state
        self.step_counter += 1
        
        done = bool("done" in received)
        info = {}
        
        reward = self.time_reward
        winner = None
        if done or "winner" in received:
            done = True
            if "winner" in received:
                winner = int(received["winner"])
                if winner != -1:
                    reward = 1 if winner == PLAYER_1 else -1
                info["winner"] = winner
                self.last_winner = winner
            else:
                self.last_winner = -1
                    
        if self.survivor and done:
            if not "winner" in info:
                reward = 1#self.step_counter / self.max_steps
            else:
                reward = -1
        
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
        if not "ending" in received and self.verbose:
            print("Something wrong with ending game", flush=True)
        self.sock.close()
        if self.game_path:
            self.game_p.wait()
        self.game_log.close()
        if self.verbose:
            print("Game", self.game_port, "closed port", flush=True)
        if self.stdout_path and self.verbose:
            sys.stdout.close()