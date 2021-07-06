from tank_env import TankEnv
from stable_baselines3 import PPO
import numpy as np
import argparse
import matplotlib.pyplot as plt
from train import load_pop, load_stats
from gt_plot import sorted_keys, avg_elo
import os
import json
from elo import elo_change
from tank_env import elo_based_choice

def new_human_stats():
    return {
        "elo":[],
        "win_rate":{}
    }

def play_match(env, num_games):  
    score = [0,0,0]
    print("Score: [Player1 Wins, Player2 Wins, Ties]")
    dummy_action = np.zeros(5, dtype=np.float32)
        
    env.reset()
    curr_game = 0
    while curr_game < num_games:
        _, _, done, info = env.step(dummy_action)
        if done:
            score[info["winner"]] += 1
            curr_game += 1
            print("Score:", score)
            if curr_game < num_games:
                env.reset()
    return score
    
def get_human_stats(human_db):
    while True:
        human_username = input("Please enter your username: ")
        if os.path.exists(human_db+human_username+".json"):
            while True:
                confirm = input("A player with this username exists, do you wish to continue? (Y/N): ")
                if confirm == 'y' or confirm == 'Y':
                    return json.load(human_db+human_username+".json")
                elif confirm == 'n' or confirm == 'N':
                    break
                else:
                    print("Input other than 'Y' or 'N' detected")
        else:
            return new_human_stats()
            
def opp_fp(model_dir, opp):
    return model_dir+opp+'/'+opp+"_0"
    
def human_matchmaking(args):
    WINS=0
    LOSSES=1
    GAMES=2
    
    pop = load_pop(args.model_dir)
    all_stats = {}
    for p in pop:
        all_stats[p] = load_stats(args.model_dir, p)
        
    all_opps = sorted_keys(all_stats)
    all_opps.reverse()
    
    all_elos = []
    for opp in all_opps:
        all_elos.append(int(avg_elo(all_stats[opp], avg_len=args.avg_len)))
        
    human_stats = get_human_stats(args.human_db)
    
    current_opp_idx = len(all_elos)//2
    current_opp = all_opps[current_opp_idx]
    current_opp_elo = all_elos[current_opp_idx]
    human_elo = human_stats["elo"][-1] if len(human_stats["elo"]) > 0 else current_opp_elo
        
    try:
        env = TankEnv(args.game_path, 
            opp_fp_and_elo=[(opp_fp(args.model_dir,current_opp), current_opp_elo)], 
            game_port=args.base_port, 
            my_port=args.my_port, 
            image_based=args.image_based,
            level_path=args.level_path,
            p=args.env_p
        )
    
        print("Starting matchmaking")
        while human_elo <= all_elos[-1]:
            print("Current opp:", current_opp)
            print("Opp elo:", current_opp_elo)
            print("Human elo:", human_elo)
            
            score = play_match(env, args.num_games)
            human_win_rate = ((score[WINS]-score[LOSSES])/sum(score)+1)/2
            K=16
            human_elo_change, _ = elo_change(human_elo, current_opp_elo, K, human_win_rate)
            human_elo += int(human_elo_change)
            
            human_stats["elo"].append(human_elo)
            if not current_opp in human_stats["win_rate"]:
                human_stats["win_rate"][current_opp] = [0,0,0]
            human_stats["win_rate"][current_opp][WINS] += score[WINS]
            human_stats["win_rate"][current_opp][LOSSES] += score[LOSSES]
            human_stats["win_rate"][current_opp][GAMES] += sum(score)
            
            D=5.
            current_opp_idx = elo_based_choice(all_elos, human_elo, D)
            current_opp = all_opps[current_opp_idx]
            current_opp_elo = all_elos[current_opp_idx]
            env.load_new_opp(0, opp_fp(args.model_dir, current_opp), current_opp_elo)
            
        print("CONGRATS, YOU ARE BETTER THAN ALL THE AGENTS!")
            
    finally:
        env.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("human_db", type=str, help="Dir path of human player database")
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base model for models, which have already been evaluated by grand_tournament.py")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games for human to play before changing agent model")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
    parser.add_argument("--my_port", type=int, default=50001, help="Port to be used on python side of network socket connection")
    parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--env_p", type=int, default=3, help="Image-based environment will draw one in-game grid square as p^2 pixels")
    parser.add_argument("--avg_len", type=int, default=5, help="Length used for calcuating average elo (of n-previous iteration elos)")
    args = parser.parse_args()
    print(args)
    
    human_matchmaking(args)