import socket
from datetime import datetime;
import time;
import json
import numpy as np
import argparse

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
parser.add_argument("--human_player_one", action='store_true',
                    help = "Indicates that player one is a human player")
parser.add_argument("--human_player_two", action="store_true",
                    help = "Indicates that player two is a human player")
parser.add_argument("--no_shoot", action="store_true",
                    help = "Flag that stops AI from shooting")
parser.add_argument("--shoot", action="store_true",
                    help = "Flag that makes AI always shoot")
parser.add_argument("--no_move", action="store_true",
                    help = "Flag that makes AI stay still (will still aim/shoot though)")
parser.add_argument("--no_aim", action="store_true",
                    help = "Flag that makes AI aim straight (relative to their movement direction")
                    
args = parser.parse_args()
print(args)

def Send(message):
    data = json.dumps(message)
    sock.sendall(bytes(data, encoding="utf-8"))
    #print("Sent:", data)
    
def Receive():
    received = sock.recv(1024)
    received = received.decode("utf-8")
    #print("Received:", received)
    return json.loads(received)

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    for _ in range(args.connect_attempts):
        try:
            sock.connect((args.game_ip, args.game_port))
            break
        except ConnectionRefusedError:
            print("Could not connect to IP", args.game_ip, "on port", args.game_port, "...sleeping for one second")
            time.sleep(1)
    print("Connected to IP", args.game_ip, "on port", args.game_port)
    
    for i in range(args.num_games):
        Send( {"start":True} )
        received = Receive()
        #starting_message = { "start":true }
        #data = json.dumps(starting_message)
        #sock.sendall(bytes(data, encoding="utf-8"))
        
        #received = sock.recv(1024)
        #received = received.decode("utf-8")
        #received = json.loads(received)
        
        if "starting" in received and received["starting"]:
            print("Starting game", i)
            start = time.time()
            while True:
                received = Receive()
                receive_counter += 1
                
                if "done" in received:
                    break
                    
                print(len(received["state"]), received["state"])
                
                message = {}
                #red_action = (np.random.rand(5) * 2 - 1).tolist()
                if not args.human_player_one:
                    message[1] = (np.random.rand(5) * 2 - 1).tolist()
                    if args.no_shoot:
                        message[1][-1] = 0
                    if args.shoot:
                        message[1][-1] = 1
                    if args.no_move:
                        message[1][0:2] = np.zeros(2)
                    if args.no_aim:
                        message[1][2:4] = np.zeros(2)
                #blue_action = np.random.rand(5) * 2 - 1
                if not args.human_player_two:
                    message[2] = (np.random.rand(5) * 2 - 1).tolist()
                    if args.no_shoot:
                        message[2][-1] = 0
                    if args.shoot:
                        message[2][-1] = 1
                    if args.no_move:
                        message[2][0:2] = np.zeros(2)
                    if args.no_aim:
                        message[2][2:4] = np.zeros(2)
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
        print("Something might have happened in Unity env")
    sock.close()
    print("\nClosed socket")