import socket
from datetime import datetime;
import time;
import json
import numpy as np

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
    host, port = "127.0.0.1", 50000
    connect_attempts = 60
    number_of_games = 3
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    for _ in range(connect_attempts):
        try:
            sock.connect((host, port))
            break
        except ConnectionRefusedError:
            print("Could not connect to IP", host, "on port", port, "...sleeping for one second")
            time.sleep(1)
    print("Connected to IP", host, "on port", port)
    
    for i in range(number_of_games):
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
            while True:
                received = Receive()
                if "done" in received:
                    break
                
                message = {}
                red_action = np.random.rand(5) * 2 - 1
                red_action[4] = 1
                message[1] = red_action.tolist()
                blue_action = np.random.rand(5) * 2 - 1
                message[2] = blue_action.tolist()
                Send(message)
                
            print("Finished game", i)
        else:
            print("Something wrong with starting game", i)
            break

    Send( {"end":True} )
    received = Receive()
    if not "ending" in received:
        print("Something might have happened in Unity env")
    sock.close()
    print("\nClosed socket")