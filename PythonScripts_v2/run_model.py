from tank_env import TankEnv
from stable_baselines3 import PPO
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("opp", type=str, help="Path to model to be used as opponent")
parser.add_argument("--p1", type=str, default=None, help="Path to model for player 1")
parser.add_argument("--p1same", action="store_true", help="Indicates that opp model should be used as p1")
parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
parser.add_argument("--my_port", type=int, default=50500, help="Port to be used on Python side of network socket connection")
args = parser.parse_args()
print(args)

env = TankEnv(None, opp_fp_and_elo=[(args.opp, 1000)], game_port=args.base_port, my_port=args.my_port)
model = None
if args.p1:
    model = PPO.load(args.p1)
elif args.p1same:
    model = PPO.load(args.opp)
    
obs = env.reset()
while True:
    if model:
        action, _ = model.predict(obs)
    else:
        action = np.zeros(5, dtype=np.float32)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()