from tank_env import TankEnv
import numpy as np
from numpy import savez_compressed
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("num_obs", type=int, help="Number of observations to generate")
parser.add_argument("desktop_game_path", type=str, help="File path of desktop game executable")
parser.add_argument("server_game_path", type=str, help="File path of server game executable")
parser.add_argument("level_path", type=str, help="File path of game level")
parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
parser.add_argument("--my_port", type=int, default=50500, help="Port to be used on Python side of network socket connection")
parser.add_argument("--save_loc", type=str, default="dataset.npz", help="File path to save data to")
args = parser.parse_args()
print(args)

obs_set = np.zeros((args.num_obs, 52), dtype=np.float32)
img_set = np.zeros((args.num_obs, 36, 60, 3), dtype=np.uint8)
try:
    env = TankEnv(args.desktop_game_path, 
        opp_fp_and_elo=[], 
        game_port=args.base_port, 
        my_port=args.my_port,
        rand_opp=True
        )
    canvas = TankEnv(args.server_game_path,
        opp_fp_and_elo=[], 
        game_port=args.base_port+1, 
        my_port=args.my_port+1, 
        image_based=True,
        level_path=args.level_path,
        rand_opp=True
        )
        
    obs = env.reset()
    for i in tqdm(range(args.num_obs)):
        # Save states
        obs_set[i] = obs.copy()
        canvas.draw_state(obs)
        img_set[i] = canvas.state.copy()
        # Generate next observation
        action = np.random.rand(5) * 2 - 1
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    savez_compressed(args.save_loc, obs=obs_set, img=img_set)
finally:
    env.close()
    canvas.close()