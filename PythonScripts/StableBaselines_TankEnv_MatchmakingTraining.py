import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
from config_gen import config_gen
import json
import subprocess

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
parser.add_argument("--idx", type=int, default=0, help="For parallel processing, portion of population this job will train (from 1 to {args.part} )")
parser.add_argument("--part", type=int, default=0, help="For parallel processing, number of partitions population is being split into")
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

if args.idx and args.part:
    start = round((len(population)/args.part*(args.idx-1)))
    end = round((len(population)/args.part*(args.idx)))
    my_pop = population[start:end]
    my_pop_elos = pop_elos[start:end]
else:
    my_pop = population
    my_pop_elos = pop_elos
    
print("My population to train:", [x for x in zip(my_pop, my_pop_elos)])

for p,p_elo in zip(my_pop, my_pop_elos):
    id = "".join(p.split('_')[:-1])
    # Generate game config for training
    config_gen(args.game_config_file_path, random_start=args.rs, port=50000+args.idx)
    # Establish opponents for model to play against
    opp_file_path = args.model_dir + id + "/opponents.txt"
    with open(opp_file_path, 'w') as opp_file:
        for opp,opp_elo in zip(population, pop_elos):
            if p == opp:
                continue
            opp_file.write(opp + "\t" + str(opp_elo) + "\n")
    # Loop forever so that if system fails, that error will keep repeating (for debugging purposes)
    # If that error only happens occasionaly, then this loop will be broken
    while True:
        # Run game
        with open(os.path.expanduser(args.gamelog), 'w') as gl:
            game_p = subprocess.Popen(args.game_path, stdout=gl, stderr=gl)
            # Execute training script
            cmd_list = ["python", args.training_script, 
                        args.model_dir, id,
                        "--steps", str(args.steps),
                        "--elo", str(p_elo),
                        "--port", str(50000+args.idx)]
            with open(os.path.expanduser(args.model_dir + id + "/train_log.txt"), 'a') as tl:
                tl.write("Starting training at " + p.split('_')[-1] + " steps by worker " + str(args.idx) + "\n")
                train_p = subprocess.Popen(cmd_list, stdout=tl, stderr=tl)
                train_return = train_p.wait()
                tl.write("Ending training with exit code: " + str(train_return) + "\n")
            game_p.kill()
        if train_return in [0, -6, -7, -11]:
            break
        else:
            print("Worker", args.idx, "had an exit code of", train_return, "so is redoing MT of", id)
    
print("Training complete")