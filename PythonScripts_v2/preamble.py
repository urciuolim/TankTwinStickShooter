from tank_env import TankEnv
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json
from random import choice, randint
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
    
def gen_name(noun_file_path, adj_file_path, model_dir):
    with open(noun_file_path, 'r') as noun_file, open(adj_file_path, 'r') as adj_file:
        while True:
            name = choice(adj_file.readlines()).strip('\n').capitalize()+\
                choice(noun_file.readlines()).strip('\n').capitalize()+str(randint(0,100))
            if not os.path.isdir(model_dir + name):
                return name
                
def init_stats():
    model_stats = {
        "num_steps":0,
        "last_eval_steps":0,
        "last_elo_change_steps":0,
        "performance": {},
        "elo": {
            0:1000
        },
        "parent": None,
        "matching_agent": None,
        "nemesis": False,
        "survivor": False
    }
    return model_stats
    
def save_new_model(name, env, batch_size, n_steps, model_dir):
    model = PPO("MlpPolicy", env, batch_size=batch_size, n_steps=n_steps)
    model.save(model_dir + name + '/' + name + "_0")
    
def save_stats_file(path, *extras):
    model_stats = init_stats()
    for extra in extras:
        model_stats[extra[0]] = extra[1]
    with open(path, 'w') as stats_file:
        json.dump(model_stats, stats_file, indent=4)
    
def gen_agent(my_env, batch_size, n_steps, model_dir, noun_file_path, adj_file_path):
    name = gen_name(noun_file_path, adj_file_path, model_dir)
    save_new_model(name, my_env, batch_size, n_steps, model_dir)
    save_stats_file(args.model_dir + name + "/stats.json")
    print("Created", name, flush=True)
    return name
    
def gen_nemesis(agent_name, my_env, batch_size, n_steps, model_dir):
    nemesis_name = agent_name + "-nemesis"
    save_new_model(nemesis_name, my_env, batch_size, n_steps, model_dir)
    save_stats_file(model_dir + nemesis_name + "/stats.json", ("nemesis", True), ("matching_agent", agent_name))
    print("Created", nemesis_name, flush=True)
    return nemesis_name
    
def gen_survivor(agent_name, my_env, batch_size, n_steps, model_dir):
    survivor_name = agent_name + "-survivor"
    save_new_model(survivor_name, my_env, batch_size, n_steps, model_dir)
    save_stats_file(model_dir + survivor_name + "/stats.json", ("survivor", True), ("matching_agent", agent_name))
    print("Created", survivor_name, flush=True)
    return survivor_name
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("pop_training_script", type=str, help="Training script path that trains entire population")
    parser.add_argument("tournament_script", type=str, help="Tournament script path")
    #parser.add_argument("replace_script", type=str, help="Agent replacement script path")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("noun_file_path", type=str, help="Path to noun file used to generate names")
    parser.add_argument("adj_file_path", type=str, help="Path to adj file used to generate names")
    parser.add_argument("--start", type=int, default=8, help="Number of starting agents to start population with")
    #parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to direct game logging to")
    parser.add_argument("--nem", action="store_true", help="Train with nemeses")
    parser.add_argument("--surv", action="store_true", help="Train with survivors")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for all agents")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port for environments")
    args = parser.parse_args()
    print(args, flush=True)
        
    # Check that all file paths needed for PBT are valid
    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not os.path.exists(args.game_path):
        raise FileNotFoundError("Inputted game path does not lead to an existing file")
    if not os.path.exists("./Assets/config.json"):
        raise FileNotFoundError("Game config file not found at ./Assets/config.json")
    if not os.path.exists(args.pop_training_script):
        raise FileNotFoundError("Python population training script not found")
    if not os.path.exists(args.tournament_script):
        raise FileNotFoundError("Python tournament script not found")
    #if not os.path.exists(args.replace_script):
        #raise FileNotFoundError("Python agent replacement script not found")
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Base directory for agent models is not a folder")
    if not os.path.exists(args.noun_file_path):
        raise FileNotFoundError("Inputted path does not lead to noun file")
    if not os.path.exists(args.adj_file_path):
        raise FileNotFoundError("Inputted path does not lead to adjective file")
            
    envs = []
    for i in range(args.num_envs):
        envs.append(
            lambda game_path=args.game_path, b=args.base_port+(i*2), c="gamelog-"+str(i)+".txt": 
                    TankEnv(game_path,
                            game_port=b,
                            game_log_path=c
                    )
        )
    env_stack = DummyVecEnv(envs)

    population = []
    for i in range(args.start):
        agent_name = gen_agent(env_stack, args.batch_size, args.batch_size//args.num_envs, args.model_dir, args.noun_file_path, args.adj_file_path)
        population.append(agent_name)
        if args.nem:
            population.append(gen_nemesis(agent_name, env_stack, args.batch_size, args.batch_size//args.num_envs, args.model_dir))
        if args.surv:
            population.append(gen_survivor(agent_name, env_stack, args.batch_size, args.batch_size//args.num_envs, args.model_dir))
    if args.start:
        with open(args.model_dir + "/population.txt", 'w') as pop_file:
            for p in population:
                pop_file.write(p + '\n')
        
    env_stack.close()
    
    print("PBT Preamble complete", flush=True)