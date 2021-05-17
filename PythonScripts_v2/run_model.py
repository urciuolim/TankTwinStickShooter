from tank_env import TankEnv
from stable_baselines3 import PPO
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("opp", type=str, help="Path to model to be used as opponent")
parser.add_argument("--p1", type=str, default=None, help="Path to model for player 1")
parser.add_argument("--p1same", action="store_true", help="Indicates that opp model should be used as p1")
parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
parser.add_argument("--my_port", type=int, default=50500, help="Port to be used on Python side of network socket connection")
parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based, and will show those states using matplotlib")
parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
parser.add_argument("--ai_view", action="store_true", help="Indicates that AI version of game state should be rendered")
args = parser.parse_args()
print(args)

env = TankEnv(None, 
    opp_fp_and_elo=[(args.opp, 1000)], 
    game_port=args.base_port, 
    my_port=args.my_port, 
    image_based=args.image_based,
    level_path=args.level_path)
model = None
if args.p1:
    model = PPO.load(args.p1)
elif args.p1same:
    model = PPO.load(args.opp)
    
obs = env.reset()
if args.image_based and args.ai_view:
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
while True:
    if args.image_based and args.ai_view:
        plt.imshow(obs, origin="lower")
        fig.canvas.draw()
    if model:
        action, _ = model.predict(obs)
    else:
        action = np.zeros(5, dtype=np.float32)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()