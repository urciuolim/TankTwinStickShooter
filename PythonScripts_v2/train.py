import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from tank_env import TankEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import json

def load_stats(model_dir, agent_id):
    with open(model_dir+agent_id+"/stats.json", 'r') as agent_stats_file:
        return json.load(agent_stats_file)
        
def save_stats(model_dir, agent_id, agent_stats):
    with open(model_dir+agent_id+"/stats.json", 'w') as agent_stats_file:
        json.dump(agent_stats, agent_stats_file, indent=4)
        
def load_pop(model_dir):
    pop = []
    with open(model_dir+"/population.txt", 'r') as pop_file:
        for line in pop_file.readlines():
            pop.append(line.strip('\n'))
    return pop
    
def last_model_path(model_dir, agent_id, agent_stats):
    return model_dir+agent_id+'/'+agent_id+'_'+str(agent_stats["last_eval_steps"])

def last_elo(agent_stats):
    return agent_stats["elo"][str(agent_stats["last_eval_steps"])]

def make_env_stack(num_envs, game_path, base_port, game_log_path, opp_fp_and_elo, trainee_elo, elo_match=True, survivor=False):
    envs = []
    for i in range(num_envs):
        envs.append(
            lambda game_path=game_path, b=base_port+(i*2), c=game_log_path.replace(".txt", "-"+str(i)+".txt"), d=opp_fp_and_elo, e=elo_match, f=trainee_elo, g=survivor: 
                    TankEnv(game_path,
                            game_port=b,
                            game_log_path=c,
                            opp_fp_and_elo=d,
                            elo_match=e,
                            center_elo=f,
                            survivor=g
                    )
        )
    env_stack = SubprocVecEnv(envs, start_method="fork")
    env_stack.reset()
    return env_stack
    
def get_opps_and_elos(model_dir, agent_id):
    pop = load_pop(model_dir)
    pop.remove(agent_id)
    
    pop_fps = []
    elos = []
    for p in pop:
        p_stats = load_stats(model_dir, p)
        pop_fps.append(last_model_path(model_dir, p, p_stats))
        elos.append(last_elo(p_stats))
    return list(zip(pop_fps, elos))
    
def train_agent(model_dir, agent_id, game_path, base_port, num_envs, num_steps):
    # Load agent and env
    agent_stats = load_stats(model_dir, agent_id)
    if not (agent_stats["nemesis"] or agent_stats["survivor"]):
        opp_fp_and_elo = get_opps_and_elos(model_dir, agent_id)
    else:
        opp_fp_and_elo = [(last_model_path(model_dir, agent_stats["matching_agent"], load_stats(model_dir, agent_stats["matching_agent"])), last_elo(agent_stats))]
        
    env_stack = make_env_stack(num_envs, game_path, base_port, model_dir+agent_id+"/gamelog.txt", opp_fp_and_elo, last_elo(agent_stats), survivor=agent_stats["survivor"])
    agent_model_path = last_model_path(model_dir, agent_id, agent_stats)
    agent = PPO.load(agent_model_path, env=env_stack)
    print("Loaded model saved at", agent_model_path, flush=True)
    # Learn
    agent.learn(total_timesteps=num_steps)
    # Save and cleanup
    env_stack.close()
    agent_stats["num_steps"] += num_steps
    new_agent_save_path = model_dir+agent_id+'/'+agent_id+'_'+str(agent_stats["num_steps"])
    agent.save(new_agent_save_path)
    print("Saved model at", new_agent_save_path, flush=True)
    save_stats(model_dir, agent_id, agent_stats)
    
def validate_args(args):
    # Load model and intialize environment
    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Base directory input is not a folder")
        
    pop = load_pop(args.model_dir)
    if "agent_id" in args and not args.agent_id in pop:
        raise ValueError("Given agent ID is not in population")
    if len(pop) <= 0:
        raise ValueError("Population in given base directory does not have any IDs")
    for p in pop:
        if not os.path.isdir(args.model_dir + p):
            raise FileNotFoundError("ID " + p + " does not have a folder in base directory")
        if not os.path.exists(args.model_dir + p + "/stats.json"):
            raise FileNotFoundError("ID " + p + " does not have a stats file")
    
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("agent_id", type=str, help="ID of agent model to be trained")
    parser.add_argument("--num_steps", type=int, default=100000, help = "Total number of steps to train for")
    parser.add_argument("--base_port", type=int, default=50000, help = "Port that environment will communicate on")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    args = parser.parse_args()
    print(args, flush=True)
    validate_args(args)
    print("Starting training of", args.agent_id, "for", args.num_steps, "steps", flush=True)
    train_agent(args.model_dir, args.agent_id, args.game_path, args.base_port, args.num_envs, args.num_steps)
    print("Training of", args.agent_id, "complete", flush=True)