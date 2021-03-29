from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import subprocess
import argparse
import json
from random import choice, randint
from stable_baselines3 import PPO
from config_gen import config_gen

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, help="File path of game executable")
parser.add_argument("game_config_file_path", type=str, help="File path of game config file")
parser.add_argument("pop_training_script", type=str, help="Training script path that trains entire population")
parser.add_argument("training_script", type=str, help="Training script path for individual agent training")
parser.add_argument("tournament_script", type=str, help="Tournament script path")
parser.add_argument("eval_script", type=str, help="Evaluation script path")
parser.add_argument("replace_script", type=str, help="Agent replacement script path")
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("pop_file_path", type=str, help="Path for file that will contain IDs of agents in population to train")
parser.add_argument("noun_file_path", type=str, help="Path to noun file used to generate names")
parser.add_argument("adj_file_path", type=str, help="Path to adj file used to generate names")
parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to direct game logging to")
parser.add_argument("--start", type=int, default=8, help="Number of starting agents to start population with")
args = parser.parse_args()
print(args)
    
if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
    args.model_dir = args.model_dir + "/"
if not os.path.exists(args.game_path):
    raise FileNotFoundError("Inputted game path does not lead to an existing file")
if not os.path.exists(args.game_config_file_path):
    raise FileNotFoundError("Game config file not found")
if not os.path.exists(args.pop_training_script):
    raise FileNotFoundError("Python population training script not found")
if not os.path.exists(args.training_script):
    raise FileNotFoundError("Python training script not found")
if not os.path.exists(args.tournament_script):
    raise FileNotFoundError("Python tournament script not found")
if not os.path.exists(args.eval_script):
    raise FileNotFoundError("Python evaluation script not found")
if not os.path.exists(args.replace_script):
    raise FileNotFoundError("Python agent replacement script not found")
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
if not os.path.exists(args.pop_file_path) and not args.start:
    raise FileNotFoundError("Inputted path does not lead to population file (and starting fresh has not been indicated)")
if not os.path.exists(args.noun_file_path):
    raise FileNotFoundError("Inputted path does not lead to noun file")
if not os.path.exists(args.adj_file_path):
    raise FileNotFoundError("Inputted path does not lead to adjective file")
    
def gen_name():
    with open(args.noun_file_path, 'r') as noun_file, open(args.adj_file_path, 'r') as adj_file:
        while True:
            name = choice(adj_file.readlines()).strip('\n').capitalize()+\
                choice(noun_file.readlines()).strip('\n').capitalize()+str(randint(0,100))
            if not os.path.isdir(args.model_dir + name):
                return name
    
config_gen(args.game_config_file_path, port=50000)
with open(os.path.expanduser(args.gamelog), 'w') as gl:
    game_p = subprocess.Popen(args.game_path, stdout=gl, stderr=gl)
env = IndvTankEnv(TankEnv())
population = []
for i in range(args.start):
    while True:
        name = gen_name()
        if not os.path.isdir(args.model_dir + name):
            break
    population.append(name)
    model = PPO("MlpPolicy", env)
    model.save(args.model_dir + name + "/" + name + "_0")
    model_stats = {
        "num_steps":0,
        "performance": {
            "avg_reward":[],
            "avg_steps":[],
            "trained_steps":[]
        },
        "elo": {
            "value": [1000],
            "steps": [0]
        },
        "parent": None
    }
    with open(args.model_dir + name + "/stats.json", 'w') as model_stats_file:
        json.dump(model_stats, model_stats_file, indent=4)
    print("Created", name)
env.close()
game_p.kill()

with open(args.pop_file_path, 'w') as pop_file:
    for p in population:
        pop_file.write(p + '\n')
    
print("PBT Preamble complete")