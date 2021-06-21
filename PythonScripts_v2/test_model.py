from tank_env import TankEnv
from stable_baselines3 import PPO
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, default=None, help="File path of game executable")
parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
parser.add_argument("--my_port", type=int, default=50500, help="Port to be used on Python side of network socket connection")
parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based, and will show those states using matplotlib")
parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
parser.add_argument("--ai_view", action="store_true", help="Indicates that AI version of game state should be rendered")
parser.add_argument("--train", action="store_true", help="Indicates that test should try training model (as opposed to just running eval)")
parser.add_argument("--num_steps", type=int, default=128, help="Number of steps to run for")
parser.add_argument("--env_p", type=int, default=3, help="p^2 pixels will represent one in-game grid square")
args = parser.parse_args()
print(args)

env = TankEnv(args.game_path, 
    opp_fp_and_elo=[], 
    game_port=args.base_port, 
    my_port=args.my_port, 
    image_based=args.image_based,
    level_path=args.level_path,
    rand_opp=True,
    p=args.env_p)
if args.image_based:
    model = PPO("CnnPolicy", env, n_steps=64)
else:
    model = PPO("MlpPolicy", env, n_steps=64)
    
print(model.policy)
  
try:
    if args.train:
        model.learn(total_timesteps=args.num_steps)
    else:
        obs = env.reset()
        if args.image_based and args.ai_view:
            fig = plt.gcf()
            fig.show()
            fig.canvas.draw()
        for _ in tqdm(range(args.num_steps)):
            if args.image_based and args.ai_view:
                plt.imshow(obs, origin="lower", interpolation='none')
                fig.canvas.draw()
            if model:
                action, _ = model.predict(obs)
            elif args.rand_p1:
                action = np.random.rand(5) * 2 - 1
            else:
                action = np.zeros(5, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
finally:
    env.close()