import socket
from datetime import datetime;
import time;
import json
import numpy as np
import argparse
from RandomAgent import RandomAgent
from AgentLogger import AgentLogger

send_counter = 0
receive_counter = 0

parser = argparse.ArgumentParser()
parser.add_argument("--game_ip", type=str, default="127.0.0.1",
                    help = "IP of game server")
parser.add_argument("--game_port", type=int, default = 50000,
                    help = "Port of game server")
parser.add_argument("--connect_attempts", type=int, default=60,
                    help = "Number of attempts to connect to game")
parser.add_argument("--num_games", type=int, default=3,
                    help = "Number of games to play")
parser.add_argument("--players", type=str, default="1|1",
                    help = "Indication of which player is which, each seperated by a '|'. 0=human, 1=random, etc.")
parser.add_argument("--log", action="store_true",
                    help = "Flag indicates that AI agents should log incoming states and outgoing actions")
parser.add_argument("--log_dir", type=str, default=".",
                    help = "Directory to store log files in")
parser.add_argument("--seed", type=int, default=None, help = "Seed to be used for np.random")
parser.add_argument("--no_shoot", action="store_true",
                    help = "Flag that stops AI from shooting")
'''
parser.add_argument("--shoot", action="store_true",
                    help = "Flag that makes AI always shoot")
parser.add_argument("--no_move", action="store_true",
                    help = "Flag that makes AI stay still (will still aim/shoot though)")
parser.add_argument("--no_aim", action="store_true",
                    help = "Flag that makes AI aim straight (relative to their movement direction")
'''
                    
args = parser.parse_args()
print(args)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def Send(message):
    global sock
    data = json.dumps(message)
    sock.sendall(bytes(data, encoding="utf-8"))
    
def Receive():
    global sock
    received = sock.recv(1024)
    received = received.decode("utf-8")
    return json.loads(received)
    
if __name__ == "__main__":
    if args.seed:
        np.random.seed(args.seed)
        seeds = np.random.randint(0, 2**32-1, size=(args.num_games,))
    agents = []
    for i,player in enumerate(args.players.split('|')):
        agent = None
        if player == '1':
            agent = RandomAgent("RandomAgent" + str(i), (52,), np.array([[-1,-1,-1,-1,0], [1,1,1,1,1]]))
        if args.log:
            agent = AgentLogger(agent, args.log_dir)
        agents.append(agent)

    for _ in range(args.connect_attempts):
        try:
            sock.connect((args.game_ip, args.game_port))
            break
        except ConnectionRefusedError:
            print("Could not connect to IP", args.game_ip, "on port", args.game_port, "...sleeping for one second")
            time.sleep(1)
    print("Connected to IP", args.game_ip, "on port", args.game_port)
    
    for i in range(args.num_games):
        if args.seed:
            np.random.seed(seeds[i])
        Send( {"start":True} )
        received = Receive()
        
        if "starting" in received and received["starting"]:
            print("Starting game", i)
            start = time.time()
            while True:
                received = Receive()
                receive_counter += 1
                
                if "done" in received:
                    break
                
                message = {}
                for a,agent in enumerate(agents, start=1):
                    action = agent.get_action(np.array(received["state"]))
                    if args.no_shoot:
                        action[-1] = 0
                    message[a] = action.tolist()
                Send(message)
                send_counter += 1
                
            print("Finished game", i)
            end = time.time()
            print("It took", end-start, "seconds")
            print(receive_counter, "states received")
            print(send_counter, "actions sent")
            receive_counter = 0
            send_counter = 0
        else:
            print("Something wrong with starting game", i)
            break

    Send( {"end":True} )
    received = Receive()
    if not "ending" in received:
        print("Something wrong with ending game")
    sock.close()
    print("\nClosed socket")