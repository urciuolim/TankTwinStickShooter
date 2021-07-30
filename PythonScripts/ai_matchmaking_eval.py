from tank_env import TankEnv
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from train import load_pop, load_stats, save_stats
from gt_plot import avg_elo
import os
from stable_baselines3.common.utils import set_random_seed
from human_matchmaking import opp_fp
import random
from elo import elo_change
from tqdm import tqdm
import sys

def curr_model_path(model_dir, agent_id, agent_stats):
    return model_dir+agent_id+'/'+agent_id+'_'+str(agent_stats["num_steps"])

def make_ai_matchmaker_eval_stack(game_path, base_port, image_based, level_path, env_p, num_envs):
        envs = []
        for i in range(num_envs):
            envs.append(
                lambda a=game_path, b=base_port+(i*2), c=base_port+(i*2)+1, d=image_based, e=level_path, f=env_p: 
                    TankEnv(a, 
                        opp_fp_and_elo=[], 
                        game_port=b, 
                        my_port=c, 
                        elo_match=False,
                        image_based=d,
                        level_path=e,
                        p=f
                    )
            )
        env_stack = SubprocVecEnv(envs, start_method="fork")
        return env_stack

def ai_matchmaking_eval(args):
    PLAYER_1=0
    PLAYER_2=1

    if args.model_dir != '/':
        args.model_dir += '/'
    pop = load_pop(args.model_dir)
    all_stats = {}
    for p in pop:
        all_stats[p] = load_stats(args.model_dir, p)
        
    all_opps = list(all_stats.keys())
    
    all_elos = []
    for opp in all_opps:
        all_elos.append(int(avg_elo(all_stats[opp], avg_len=args.avg_len)))
          
    if args.agent_dir[-1] != '/':
        args.agent_dir += '/'
    if not os.path.exists(args.agent_dir):
        os.mkdir(args.agent_dir)
        
    if ".txt" in args.agent_id:
        with open(args.agent_dir + args.agent_id, 'r') as name_file:
            args.agent_id = name_file.readlines()[0]
    
    agent_stats = load_stats(args.agent_dir, args.agent_id)
    agent = PPO.load(curr_model_path(args.agent_dir, args.agent_id, agent_stats))
    image_based = agent_stats["image_based"]
    env_p = agent_stats["env_p"]
    
    args.num_envs = min(args.num_envs, len(all_opps))
    env_idx_to_opp_idx = [0 for _ in range(args.num_envs)]
        
    try:
        env_stack = make_ai_matchmaker_eval_stack(args.game_path, args.base_port, image_based, args.level_path, env_p, args.num_envs)
        
        for n in range(args.N_games):
            print("Agent Elo before iteration", n, ':', agent_stats["elo"]["value"][-1], flush=True)
            elo_delta = 0.
            for i in range(args.num_envs):
                env_stack.env_method("load_new_opp", 0, opp_fp(args.model_dir, all_opps[i]), 0, indices=[i])
                env_idx_to_opp_idx[i] = i
            del i
            
            states = env_stack.reset()
            envs_done = []
            next_i = args.num_envs
            prog_bar = tqdm(range(len(all_opps)), file=sys.stdout)
            
            while not all([tmp == -1 for tmp in env_idx_to_opp_idx]):
                #print("Worker", args.seed, "current opps:", env_idx_to_opp_idx, flush=True)
                reset_states = env_stack.env_method("reset", indices = envs_done)
                for state,env_idx in zip(reset_states, envs_done):
                    states[env_idx] = state
                envs_done = []
                while len(envs_done) < 1:
                    actions, _ = agent.predict(states)
                    states, _, dones, infos = env_stack.step(actions)
                    if any(dones):
                        for j,done in enumerate(dones):
                            if done:
                                # Record elo change if needed
                                if env_idx_to_opp_idx[j] != -1:
                                    win_rate = .5
                                    if "winner" in infos[j]:
                                        if infos[j]["winner"] == PLAYER_1:
                                            win_rate = 1.
                                        elif infos[j]["winner"] == PLAYER_2:
                                            win_rate = 0.
                                    elo_delta += elo_change(agent_stats["elo"]["value"][-1], all_elos[env_idx_to_opp_idx[j]], args.K, win_rate)[0]
                                    prog_bar.update()
                                # Load next opponent if needed
                                if next_i < len(all_opps):
                                    env_stack.env_method("load_new_opp", 0, opp_fp(args.model_dir, all_opps[next_i]), 0, indices=[j])
                                    env_idx_to_opp_idx[j] = next_i
                                    next_i += 1
                                else:
                                    env_idx_to_opp_idx[j] = -1
                                envs_done.append(j)
                                
            prog_bar.close()
            agent_stats["elo"]["value"][-1] += int(elo_delta)
        print("Final agent Elo:", agent_stats["elo"]["value"][-1], flush=True)
        save_stats(args.agent_dir, args.agent_id, agent_stats)
            
    finally:
        env_stack.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_dir", type=str, help="Dir path of ai agents (not in model_dir)")
    parser.add_argument("agent_id", type=str, help="ID of agent to continue training (can also be file name in agent_dir containing the ID).")
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base model for models, which have already been evaluated by grand_tournament.py")
    parser.add_argument("--N_games", type=int, default=5, help="Number of games to play against each opponent")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for all RNGs (in python)")
    parser.add_argument("--avg_len", type=int, default=5, help="Length used for calcuating average elo (of n-previous iteration elos)")
    parser.add_argument("--K", type=int, default=4, help="K parameter to be used in Elo change calculations")
    args = parser.parse_args()
    print(args, flush=True)
    
    set_random_seed(args.seed)
    ai_matchmaking_eval(args)