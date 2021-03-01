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
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("pop_file_path", type=str, help="Path to file that contains IDs of agents in population to train")
parser.add_argument("--steps", type=int, default=100000, help = "Number of steps to train each agent for")
parser.add_argument("--rs", action="store_true", help="Indicates random start locations to be used during training")
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
    
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
    
if not os.path.exists(args.pop_file_path):
    raise FileNotFoundError("Inputted path does not lead to population file")
    
population = []
with open(args.pop_file_path, 'r') as pop_file:
    for line in pop_file.readlines():
        population.append(line.strip('\n'))
        
pop_elos = []
for i,p in enumerate(population):
    if not os.path.isdir(args.model_dir + p):
        raise FileNotFoundError("Agent ID {" + p + "} does not lead to a valid model directory")
    p_stats_path = args.model_dir + p + "/stats.json"
    if not os.path.exists(p_stats_path):
        raise FileNotFoundError("Agent ID (" + p + ") does not have a stats file")
    with open(p_stats_path, 'r') as p_stats_file:
        p_stats = json.load(p_stats_file)
    p_step = p_stats["num_steps"]
    p_elo = p_stats["elo"]["value"][-1]
    if not os.path.exists(args.model_dir + p + "/" + p + "_" + str(p_step) + ".zip"):
        raise FileNotFoundError("Agent ID (" + p + ") does not have a correct saved policy file")
    population[i] += "_" + str(p_step)
    pop_elos.append(p_elo)
      
print("Training population:", [x for x in zip(population, pop_elos)])


for p,p_elo in zip(population, pop_elos):
    id = "".join(p.split('_')[:-1])
    # Setup game for training
    config_gen(args.game_config_file_path, random_start=args.rs)
    os.system(args.game_path + " > " + args.gamelog + " &")
    # Establish opponents for model to play against
    opp_file_path = args.model_dir + id + "/opponents.txt"
    with open(opp_file_path, 'w') as opp_file:
        for opp,opp_elo in zip(population, pop_elos):
            if p == opp:
                continue
            opp_file.write(opp + "\t" + str(opp_elo) + "\n")
    # Execute training script
    os.system("python " + args.training_script + " " + args.model_dir + " " + id + " --steps " + str(args.steps) + " --elo " + str(p_elo))
    
print("Training complete")