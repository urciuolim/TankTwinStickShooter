import os
import argparse
import time
from config_gen import config_gen
import json

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, help="File path of game executable")
parser.add_argument("game_config_file_path", type=str, help="File path of game config file")
parser.add_argument("training_script", type=str, help="Training script path")
parser.add_argument("eval_script", type=str, help="Evaluation script path")
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("id", type=str, help="ID of agent model to be trained")
parser.add_argument("--steps", type=int, default=10000000, help = "Total number of steps to train for")
parser.add_argument("--intervals", type=int, default=20, help="Number of intervals to split training into")
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials to do during evaluation")
parser.add_argument("--rs", action="store_true", help="Indicates random start locations to be used during training")
parser.add_argument("--buf_size", type=int, default=1, help="Size of buffer used to draw old policies from")
parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to direct game logging to")
args = parser.parse_args()
print(args)
    
if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
    args.model_dir = args.model_dir + "/"
    
if not os.path.exists(args.game_path):
    raise FileNotFoundError("Inputted game path does not lead to an existing file")
    
if not os.path.exists(args.game_config_file_path):
    raise FileNotFoundError("Game config file not found")
    
if not os.path.exists(args.training_script):
    raise FileNotFoundError("Python training file not found")
    
if not os.path.exists(args.eval_script):
    raise FileNotFoundError("Python evaluation file not found")
    
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
    
if not os.path.isdir(args.model_dir + args.id):
    raise FileNotFoundError("Inputted ID does not lead to a valid model directory")
    
curr_step = 0
stats_file_path = args.model_dir + args.id + "/stats.json"
if os.path.exists(stats_file_path):
    with open(stats_file_path, 'r') as stats_file:
        model_stats = json.load(stats_file)
        curr_step = model_stats["num_steps"]
    
step = args.steps // args.intervals
opp_file_path = args.model_dir + args.id + "/opponents.txt"

for _ in range(args.intervals):
    # Setup game for training
    config_gen(args.game_config_file_path, random_start=args.rs)
    os.system(args.game_path + " > " + args.gamelog + " &")
    # Establish opponents for model to play against
    with open(opp_file_path, 'w') as opp_file:
        for i in range(max(curr_step-(step*(args.buf_size-1)), 0), curr_step+1, step):
            opp_file.write(args.id + "_" + str(i) + "\n")
    # Execute training script
    os.system("python " + args.training_script + " " + args.model_dir + " " + args.id + " --steps " + str(step))
    curr_step += step
    # Setup game for evaluation
    config_gen(args.game_config_file_path, random_start=False)
    os.system(args.game_path + " > " + args.gamelog + " &")
    # Execute evaluation script
    os.system("python " + args.eval_script + " " + args.model_dir + " " + args.id + " --num_trials " + str(args.num_trials))
    
print("Training complete")