from tank_env import TankEnv
import numpy as np
import argparse
from tqdm import tqdm
import os
from train import load_pop, load_stats
from train_pop import subset_pop
from eval import curr_model_path
from stable_baselines3 import PPO
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Base directory for models to gather trajectories of.")
    parser.add_argument("local_pop_dir", type=str, help="Base directory for agent models (saved on local host)")
    parser.add_argument("N", type=int, help="Number of trajectories to generate")
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("--level_path", type=str, default=None, help="File path of game level")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
    parser.add_argument("--save_name", type=str, default="traj_dataset.npz", help="Name for file to save a single model's data to")
    parser.add_argument("--max_len", type=int, default=300, help="Max length of any trajectory.")
    parser.add_argument("--worker_idx", type=int, default=1, help="Index of worker (for parallel training)")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of workers (for parallel training)")
    parser.add_argument("--from_right", action="store_true", help="Indicates that data will be collected where player 1 starts on the right")
    args = parser.parse_args()
    print(args)

    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not (args.local_pop_dir[-1] == '/' or args.local_pop_dir[-1] == '\\'):
        args.local_pop_dir = args.local_pop_dir + "/"
        
    print("Worker", args.worker_idx, "got here", 1, flush=True)

    pop = load_pop(args.local_pop_dir)
    pop_stats = []
    for opp in pop:
        pop_stats.append(load_stats(args.local_pop_dir, opp))
    my_pop = subset_pop(pop, args.worker_idx, args.total_workers)
    
    print("Worker", args.worker_idx, "got here", 2, flush=True)

    for port,p in enumerate(my_pop):
        p_idx = pop.index(p)
        p_model = PPO.load(curr_model_path(args.local_pop_dir, p, pop_stats[pop.index(p)]))
        traj_set = np.full((len(pop), args.N, args.max_len+1, 12*pop_stats[p_idx]["env_p"], 20*pop_stats[p_idx]["env_p"], 3), 255, dtype=np.uint8)
        info_set = np.full((len(pop), args.N), -1, dtype=np.int16)
        
        print("Worker", args.worker_idx, "got here", 3, flush=True)
        
        try:
            env = TankEnv(args.game_path, 
                opp_fp_and_elo=[], 
                game_port=args.base_port+port, 
                my_port=args.base_port+port+1,
                level_path=args.level_path,
                image_based=pop_stats[p_idx]["image_based"],
                p=pop_stats[p_idx]["env_p"],
                verbose=True
                )
                
            print("Worker", args.worker_idx, "got here", 4, flush=True)
                
            for i,opp in enumerate(tqdm(pop, file=sys.stdout)):
                env.load_new_opp(0, curr_model_path(args.local_pop_dir, opp, pop_stats[pop.index(opp)]), 0)
                for j in range(args.N):
                    obs = env.reset()
                    side = -1 if args.from_right else 1
                    while env.raw_state[0] * side > 0:
                        obs = env.reset()
                        
                    done=False
                    for k in range(args.max_len+1):
                        traj_set[i,j,k,:,:,:] = obs
                        if done or k==args.max_len:
                            info_set[i,j] = k
                            break
                        else:
                            action, _ = p_model.predict(obs)
                            obs,_,done,_ = env.step(action)

            np.savez_compressed(args.model_dir + p + "/" + args.save_name, traj=traj_set, info=info_set)
        finally:
            env.close()