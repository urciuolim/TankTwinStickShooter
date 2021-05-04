from tank_env import TankEnv
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json
from random import choice, randint
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import numpy as np

# https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe#:~:text=PPO%20is%20a%20policy%20gradients%20method%20that%20makes,box%20on%20a%20wide%20variety%20of%20RL%20tasks.
HYPERPARAM_RANGES = {
    "n_steps": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    "batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "n_epochs": [3, 6, 10, 12, 15, 18, 21, 24, 27, 30],
    "clip_range": [0.1, 0.2, 0.3],
    "gamma": [.8, .9, .99, .999, .9999],
    "gae_lambda": [.9, .95, 1.],
    "vf_coef": [.5, 1.],
    "ent_coef": [0., .01],
    "learning_rate": [.000005, .00001, .00003, .00005, .0001, .0003, .0005, .001, .003]
}
    
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
    
def choose_hyperp(hyperp, default_idx):
    idx = default_idx + np.random.choice([-1, 0, 1], p=[.1, .8, .1])
    if idx < 0:
        idx = 0
    elif idx >= len(HYPERPARAM_RANGES[hyperp]):
        idx = len(HYPERPARAM_RANGES[hyperp])-1
    return HYPERPARAM_RANGES[hyperp][idx]
    
def save_new_model(name, env, num_envs, model_dir, batch_size=None, n_steps=None,
        n_epochs=None, clip_range=None, gamma=None, gae_lambda=None, vf_coef=None,
        ent_coef=None, learning_rate=None):
    if not batch_size:
        batch_size = choose_hyperp("batch_size", 10)
    if not n_steps:
        n_steps = max(batch_size, choose_hyperp("n_steps", 10))//num_envs
    if not n_epochs:
        n_epochs = choose_hyperp("n_epochs", 2)
    if not clip_range:
        clip_range = choose_hyperp("clip_range", 1)
    if not gamma:
        gamma = choose_hyperp("gamma", 2)
    if not gae_lambda:
        gae_lambda = choose_hyperp("gae_lambda", 1)
    if not vf_coef:
        vf_coef = choose_hyperp("vf_coef", 0)
    if not ent_coef:
        ent_coef = choose_hyperp("ent_coef", 0)
    if not learning_rate:
        learning_rate = choose_hyperp("learning_rate", 5)
    
    model = PPO("MlpPolicy", env, batch_size=batch_size, n_steps=n_steps, 
                n_epochs=n_epochs, clip_range=clip_range, gamma=gamma, gae_lambda=gae_lambda,
                vf_coef=vf_coef, ent_coef=ent_coef, learning_rate=learning_rate)
    model.save(model_dir + name + '/' + name + "_0")
    return model
    
def save_new_stats_file(path, *extras, starting_elo=None):
    model_stats = init_stats()
    for extra in extras:
        model_stats[extra[0]] = extra[1]
    if starting_elo:
        model_stats["elo"][0] = starting_elo
    with open(path, 'w') as stats_file:
        json.dump(model_stats, stats_file, indent=4)
    
def gen_agent(my_env, num_envs, model_dir, noun_file_path, adj_file_path, batch_size=None):
    name = gen_name(noun_file_path, adj_file_path, model_dir)
    agent = save_new_model(name, my_env, num_envs, model_dir, batch_size=batch_size)
    save_new_stats_file(args.model_dir + name + "/stats.json")
    print("Created", name, flush=True)
    return (name, agent)
    
def gen_nemesis(agent_name, agent, my_env, num_envs, model_dir):
    nemesis_name = agent_name + "-nemesis"
    save_new_model(nemesis_name, my_env, num_envs, model_dir, batch_size=agent.batch_size, n_steps=agent.n_steps,
        n_epochs=agent.n_epochs, clip_range=agent.clip_range(0), gamma=agent.gamma, gae_lambda=agent.gae_lambda,
        vf_coef=agent.vf_coef, ent_coef=agent.ent_coef, learning_rate=agent.learning_rate)
    save_new_stats_file(model_dir + nemesis_name + "/stats.json", ("nemesis", True), ("matching_agent", agent_name))
    print("Created", nemesis_name, flush=True)
    return nemesis_name
    
def gen_survivor(agent_name, agent, my_env, num_envs, model_dir):
    survivor_name = agent_name + "-survivor"
    save_new_model(survivor_name, my_env, num_envs, model_dir, batch_size=agent.batch_size, n_steps=agent.n_steps,
        n_epochs=agent.n_epochs, clip_range=agent.clip_range(0), gamma=agent.gamma, gae_lambda=agent.gae_lambda,
        vf_coef=agent.vf_coef, ent_coef=agent.ent_coef, learning_rate=agent.learning_rate)
    save_new_stats_file(model_dir + survivor_name + "/stats.json", ("survivor", True), ("matching_agent", agent_name))
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
    parser.add_argument("--batch_size", type=int, default=None, help="Indicates same batch size for all agents")
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
        agent_name, agent = gen_agent(env_stack, args.num_envs, args.model_dir, args.noun_file_path, args.adj_file_path, batch_size=args.batch_size)
        population.append(agent_name)
        if args.nem:
            population.append(gen_nemesis(agent_name, agent, env_stack, args.num_envs, args.model_dir))
        if args.surv:
            population.append(gen_survivor(agent_name, agent, env_stack, args.num_envs, args.model_dir))
    if args.start:
        with open(args.model_dir + "/population.txt", 'w') as pop_file:
            for p in population:
                pop_file.write(p + '\n')
        
    env_stack.close()
    
    print("PBT Preamble complete", flush=True)