import os
import argparse
import time
from config_gen import config_gen
import json
from elo import *

def get_steps(stats_file_path):
    with open(stats_file_path, 'r') as stats_file:
        return json.load(stats_file)["num_steps"]
    
def safe_get_elo(model_stats):
    if not "elo" in model_stats:
        model_stats["elo"] = {}
        model_stats["elo"]["value"] = [1000]
        model_stats["elo"]["steps"] = [model_stats["num_steps"]]
    return model_stats["elo"]["value"][-1]
    
def get_reward_clear_eval(model_stats):
    last_reward = model_stats["performance"]["avg_reward"][-1]
    model_stats["performance"]["avg_reward"] = model_stats["performance"]["avg_reward"][:-1]
    model_stats["performance"]["avg_steps"] = model_stats["performance"]["avg_steps"][:-1]
    model_stats["performance"]["trained_steps"] = model_stats["performance"]["trained_steps"][:-1]
    return (last_reward + 1.) / 2.

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, help="File path of game executable")
parser.add_argument("game_config_file_path", type=str, help="File path of game config file")
parser.add_argument("eval_script", type=str, help="Evaluation script path")
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("comp_file_path", type=str, help="File listing all competitors for tournament")
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials for each pair of competitors to play out")
parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to direct game logging to")
args = parser.parse_args()
print(args)
    
if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
    args.model_dir = args.model_dir + "/"
    
if not os.path.exists(args.game_path):
    raise FileNotFoundError("Inputted game path does not lead to an existing file")
    
if not os.path.exists(args.game_config_file_path):
    raise FileNotFoundError("Game config file not found")
    
if not os.path.exists(args.eval_script):
    raise FileNotFoundError("Python evaluation file not found")
    
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
    
if not os.path.exists(args.comp_file_path):
    raise FileNotFoundError("Competitors file not found")
    
competitors = []
with open(args.comp_file_path, 'r') as comp_file:
    for line in comp_file.readlines():
        competitors.append(line.strip('\n'))

for c in competitors:
    if not os.path.isdir(args.model_dir + c):
        raise FileNotFoundError("Competitor ID {" + c + "} does not lead to a valid model directory")
    if not os.path.exists(args.model_dir + c + "/stats.json"):
        raise FileNotFoundError("Competitor ID (" + c + ") does not have a stats file")
        
elo_changes = [0 for _ in range(len(competitors))]
# Each competitor will play each other <args.num_trials> times
for i,c in enumerate(competitors):
    for j,opp in enumerate(competitors):
        if c == opp: # Doesn't make sense to play itself as ELO scores are being adjusted
            continue
            
        print("elo_changes:", elo_changes)
        c_stats_file_path = args.model_dir + c + "/stats.json"
        opp_stats_file_path = args.model_dir + opp + "/stats.json"
        opp_file_path = args.model_dir + c + "/opponents.txt"
        
        c_steps = get_steps(c_stats_file_path)
        opp_steps = get_steps(opp_stats_file_path)
                
        print(c+ "_" + str(c_steps), "vs", opp + "_" + str(opp_steps))

        # Establish opponents for model to play against
        with open(opp_file_path, 'w') as opp_file:
            opp_file.write(opp + "_" + str(opp_steps))
            
        # Setup game for evaluation
        config_gen(args.game_config_file_path, random_start=False)
        os.system(args.game_path + " > " + args.gamelog + " &")
        # Execute evaluation script
        os.system("python " + args.eval_script + " " + args.model_dir + " " + c + " --num_trials " + str(args.num_trials))
        
        with open(c_stats_file_path, 'r') as c_stats_file:
            c_stats = json.load(c_stats_file)
            
        with open(opp_stats_file_path, 'r') as opp_stats_file:
            opp_stats = json.load(opp_stats_file)
            
        c_avg_reward = get_reward_clear_eval(c_stats)
        c_elo = safe_get_elo(c_stats)
        opp_elo = safe_get_elo(opp_stats)
        K = 32
        c_elo_change, opp_elo_change = elo_change(c_elo, opp_elo, K, c_avg_reward)
        elo_changes[i] += c_elo_change
        elo_changes[j] += opp_elo_change
        
        with open(c_stats_file_path, 'w') as c_stats_file:
            json.dump(c_stats, c_stats_file, indent=4)
            
for elo_change,c in zip(elo_changes, competitors):
    c_stats_file_path = args.model_dir + c + "/stats.json"
    with open(c_stats_file_path, 'r') as c_stats_file:
        c_stats = json.load(c_stats_file)
    c_stats["elo"]["value"].append(safe_get_elo(c_stats) + elo_change)
    c_stats["elo"]["steps"] = [c_stats["num_steps"]]
    with open(c_stats_file_path, 'w') as c_stats_file:
        json.dump(c_stats, c_stats_file, indent=4)
    
print("Tournament complete")