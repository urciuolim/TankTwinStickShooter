from ai_matchmaker import AIMatchmaker
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from train import load_pop, load_stats, save_stats
from gt_plot import sorted_keys, avg_elo
from preamble import gen_name, save_new_model
import os
import numpy as np
from stable_baselines3.common.utils import set_random_seed

def make_ai_matchmaker_stack(all_stats, all_opps, all_elos, game_path, model_dir,
        base_port=50000, image_based=False, level_path=None, env_p=3, starting_elo=None, 
        K=16, D=5., time_reward=-0.003, num_envs=1, matchmaking_mode=0, win_loss_ratio="0:0"):
        
        envs = []
        for i in range(num_envs):
            envs.append(
                lambda a=all_stats, b=all_opps, c=all_elos, d=game_path, e=model_dir, f=base_port+(i*2), g=base_port+(i*2)+1, \
                h=image_based, i=level_path, j=env_p, k=starting_elo, l=time_reward, m=matchmaking_mode, \
                n=[int(x) for x in win_loss_ratio.split(':')]: 
                        AIMatchmaker(a,b,c,d,e,
                                base_port=f,
                                my_port=g,
                                image_based=h,
                                level_path=i,
                                env_p=j,
                                starting_elo=k,
                                time_reward=l,
                                matchmaking_mode=m,
                                win_loss_ratio=n
                        )
            )
        env_stack = SubprocVecEnv(envs, start_method="fork")
        env_stack.reset()
        return env_stack
        
def init_agent(args):
    agent_name = gen_name(args.noun_file_path, args.adj_file_path, args.agent_dir)
    agent_stats = {
        "num_steps":0,
        "elo": {
            "steps": [],
            "value": []
        },
        "image_based": args.image_based,
        "env_p": args.env_p
    }
    return agent_name, agent_stats

def ai_matchmaking(args):
    set_random_seed(args.seed)

    if args.model_dir != '/':
        args.model_dir += '/'
    pop = load_pop(args.model_dir)
    all_stats = {}
    for p in pop:
        all_stats[p] = load_stats(args.model_dir, p)
        
    all_opps = sorted_keys(all_stats)
    all_opps.reverse()
    
    all_elos = []
    for opp in all_opps:
        all_elos.append(int(avg_elo(all_stats[opp], avg_len=args.avg_len)))
          
    if args.agent_dir[-1] != '/':
        args.agent_dir += '/'
    if not os.path.exists(args.agent_dir):
        os.mkdir(args.agent_dir)
        
    load_agent = True
    if args.agent_id:
        if ".txt" in args.agent_id and os.path.exists(args.agent_dir + args.agent_id):
            with open(args.agent_dir + args.agent_id, 'r') as name_file:
                agent_name = name_file.readlines()[0]
            agent_stats = load_stats(args.agent_dir, agent_name)
        elif not ".txt" in args.agent_id:
            agent_name = args.agent_id
            agent_stats = load_stats(args.agent_dir, agent_name)
        else:
            agent_name, agent_stats = init_agent(args)
            load_agent = False
    else:
        agent_name, agent_stats = init_agent(args)
        load_agent = False
        
    try:
        env_stack = make_ai_matchmaker_stack(
            all_stats,
            all_opps,
            all_elos,
            args.game_path, 
            args.model_dir,
            base_port=args.base_port,
            image_based=args.image_based,
            level_path=args.level_path,
            env_p=args.env_p,
            time_reward=args.time_reward,
            num_envs=args.num_envs,
            matchmaking_mode=args.mm,
            win_loss_ratio=args.win_loss_ratio
        )
        
        v = 1 if args.verbose else 0
        if load_agent:
            agent = PPO.load(args.agent_dir + agent_name + '/' + agent_name + '_' + str(agent_stats["num_steps"]), env=env_stack)
        else:
            agent = save_new_model(agent_name, env_stack, args.num_envs, args.agent_dir, image_based=args.image_based, image_pretrain=args.image_pretrain, verbose=v, w=args.w)
        agent.learn(total_timesteps=args.num_steps)
        
        agent_stats["num_steps"] += args.num_steps
        agent_base = args.agent_dir+agent_name+'/'
        agent_save_path = agent_base + agent_name + '_' + str(agent_stats["num_steps"])
        os.system("zip " + agent_base+"archive.zip " + agent_base+"*_*.zip")
        os.system("rm " + agent_base+"*_*.zip")
        agent.save(agent_save_path)
        
        agent_avg_elo = env_stack.env_method("get_agent_elo")
        agent_avg_elo = sum(agent_avg_elo) / len(agent_avg_elo)
        
        agent_stats["elo"]["steps"].append(agent_stats["num_steps"])
        agent_stats["elo"]["value"].append(agent_avg_elo)
        
        save_stats(args.agent_dir, agent_name, agent_stats)
        if not load_agent:
            with open(args.agent_dir + "name_" + str(args.seed) + ".txt", 'w') as name_file:
                name_file.write(agent_name)
            
    finally:
        env_stack.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_dir", type=str, help="Dir path of ai agents (not in model_dir)")
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base model for models, which have already been evaluated by grand_tournament.py")
    parser.add_argument("noun_file_path", type=str, help="Path to noun file used to generate names")
    parser.add_argument("adj_file_path", type=str, help="Path to adj file used to generate names")
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of steps for agent to train for")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
    parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--env_p", type=int, default=3, help="Image-based environment will draw one in-game grid square as p^2 pixels")
    parser.add_argument("--avg_len", type=int, default=5, help="Length used for calcuating average elo (of n-previous iteration elos)")
    parser.add_argument("--image_pretrain", type=str, default=None, help="Path to pretrained weights for cnn (without _cnn.pth or _linear.pth). Example, ./example be the path used for files saved at ./example_cnn.pth and ./example_linear.pth")
    parser.add_argument("--time_reward", type=float, default=-0.003, help="Reward (or penalty) to give agent at each timestep")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--verbose", action="store_true", help="Indicates that agent should train with verbose output.")
    parser.add_argument("--w", type=float, default=0., help="Hyperparameters used for agent will have a 'w' chance of being adjusted up or down from their default values. w=0 means stable-baselines3 default PPO values will be used.")
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for all RNGs (in python)")
    parser.add_argument("--mm", type=int, default=0, help="Matchmaking mode. 0=random, 1=ELO based")
    parser.add_argument("--win_loss_ratio", type=str, default="0:0", help="Win loss ratio to be considered (not guarenteed) when matchmaking. 0:X means only losses, X:0 means only wins. 0:0 turns this option off.")
    parser.add_argument("--agent_id", type=str, default=None, help="ID of agent to continue training (can also be file name in agent_dir containing the ID).")
    args = parser.parse_args()
    print(args, flush=True)
    
    ai_matchmaking(args)