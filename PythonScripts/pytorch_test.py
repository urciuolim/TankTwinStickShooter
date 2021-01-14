import socket
from datetime import datetime;
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    host, port = "127.0.0.1", 50000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    red = TestModel()
    blue = TestModel()
    state = None

    try:
        sock.connect((host, port))
        while True:
                received = sock.recv(1024)
                received = received.decode("utf-8")
                received = json.loads(received)
                print("Received:", received)
                if "done" in received and received["done"]:
                    break
                
                state = torch.tensor(received["state"])
                message = {}
                message[1] = red(state).tolist()
                state = torch.cat([state[6:], state[:6]])
                message[2] = blue(state).tolist()
                data = json.dumps(message)
                sock.sendall(bytes(data, encoding="utf-8"))
                print("Sent:", data)
    finally:
        sock.close()
        print("\nClosed socket")