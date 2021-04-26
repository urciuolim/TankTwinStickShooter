from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import subprocess
import argparse
import json
from random import choice, randint
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config_gen import config_gen
import time

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
parser.add_argument("--nem", action="store_true", help="Train with nemeses")
parser.add_argument("--surv", action="store_true", help="Train with survivors")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for all agents")
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
                
def init_stats():
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
        "parent": None,
        "matching_agent": None,
        "nemesis": False,
        "survivor": False
    }
    return model_stats
    
def gen_agent(my_env):
    while True:
        name = gen_name()
        if not os.path.isdir(args.model_dir + name):
            break
    model = PPO("MlpPolicy", my_env, batch_size=args.batch_size, n_steps=args.batch_size//args.num_envs)
    model.save(args.model_dir + name + "/" + name + "_0")
    model_stats = init_stats()
    with open(args.model_dir + name + "/stats.json", 'w') as model_stats_file:
        json.dump(model_stats, model_stats_file, indent=4)
    print("Created", name)
    return name
    
def gen_nemisis(agent_name, my_env):
    nemisis_name = agent_name + "-nemesis"
    nemesis = PPO("MlpPolicy", my_env, batch_size=args.batch_size, n_steps=args.batch_size//args.num_envs)
    nemesis.save(args.model_dir + nemisis_name + "/" + nemisis_name + "_0")
    nemisis_stats = init_stats()
    nemisis_stats["nemesis"] = True
    nemisis_stats["matching_agent"] = agent_name
    with open(args.model_dir + nemisis_name + "/stats.json", 'w') as nemisis_stats_file:
        json.dump(nemisis_stats, nemisis_stats_file, indent=4)
    print("Created", nemisis_name)
    return nemisis_name
    
def gen_survivor(agent_name, my_env):
    survivor_name = agent_name + "-survivor"
    survivor = PPO("MlpPolicy", my_env, batch_size=args.batch_size, n_steps=args.batch_size//args.num_envs)
    survivor.save(args.model_dir + survivor_name + "/" + survivor_name + "_0")
    survivor_stats = init_stats()
    survivor_stats["survivor"] = True
    survivor_stats["matching_agent"] = agent_name
    with open(args.model_dir + survivor_name + "/stats.json", 'w') as survivor_stats_file:
        json.dump(survivor_stats, survivor_stats_file, indent=4)
    print("Created", survivor_name)
    return survivor_name
            
game_ps = []
envs = []
with open(os.path.expanduser(args.gamelog), 'w') as gl:
    for i in range(args.num_envs):
        game_cmd_list = [args.game_path, str(50000+i)]
        #config_gen(args.game_config_file_path, port=50000+i)
        game_p = subprocess.Popen(game_cmd_list, stdout=gl, stderr=gl)
        game_ps.append(game_p)
        #time.sleep(1)
        env = IndvTankEnv(TankEnv(game_port=50000+i))
        envs.append(env)
        #input("Press enter")
        
    d = DummyVecEnv([lambda: x for x in envs])

    population = []
    for i in range(args.start):
        agent_name = gen_agent(d)
        population.append(agent_name)
        if args.nem:
            population.append(gen_nemisis(agent_name, d))
        if args.surv:
            population.append(gen_survivor(agent_name, d))
    with open(args.pop_file_path, 'w') as pop_file:
        for p in population:
            pop_file.write(p + '\n')
        
    for env in envs:
        env.close()
    for game_p in game_ps:
        game_p.kill()
    
print("PBT Preamble complete")