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
    agent_stats_file = open(model_dir+agent_id+"/stats.json", 'w')
    json.dump(agent_stats, agent_stats_file, indent=4)
    agent_stats_file.close()
        
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

def make_env_stack(num_envs, game_path, base_port, game_log_path, opp_fp_and_elo, trainee_elo, elo_match=True, survivor=False, stdout_path=None, level_path=None, image_based=False, time_reward=0., env_p=3):
    if num_envs >= 1:
        envs = []
        for i in range(num_envs):
            envs.append(
                lambda game_path=game_path, b=base_port+(i*2), c=game_log_path.replace(".txt", "-"+str(i)+".txt"), d=opp_fp_and_elo, e=elo_match, f=trainee_elo, g=survivor, h=stdout_path.replace(".txt", "-"+str(i)+".txt"), i=level_path, j=image_based, k=time_reward: 
                        TankEnv(game_path,
                                game_port=b,
                                game_log_path=c,
                                opp_fp_and_elo=d,
                                elo_match=e,
                                center_elo=f,
                                survivor=g,
                                stdout_path=h,
                                verbose=True,
                                level_path=i,
                                image_based=j,
                                time_reward=k,
                                p=env_p
                        )
            )
        if num_envs == 1:
            env_stack = SubprocVecEnv(envs, start_method="fork")
        else:
            env_stack = SubprocVecEnv(envs, start_method="forkserver")
        env_stack.reset()
        return env_stack
    else:
        env = TankEnv(game_path, game_port=base_port, game_log_path=game_log_path, opp_fp_and_elo=opp_fp_and_elo, elo_match=elo_match,
            center_elo=trainee_elo, survivor=survivor, stdout_path=stdout_path, level_path=level_path, image_based=image_based, time_reward=time_reward, p=env_p)
        env.reset()
        return env
    
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
    
def train_agent(model_dir, local_pop_dir, agent_id, game_path, base_port, num_envs, num_steps, level_path=None, time_reward=0.):
    # Load agent and env
    agent_stats = load_stats(model_dir, agent_id)
    if not (agent_stats["nemesis"] or agent_stats["survivor"]):
        opp_fp_and_elo = get_opps_and_elos(local_pop_dir, agent_id)
    else:
        opp_fp_and_elo = [(last_model_path(local_pop_dir, agent_stats["matching_agent"], load_stats(local_pop_dir, agent_stats["matching_agent"])), last_elo(agent_stats))]
        
    env_p = agent_stats["env_p"] if "env_p" in agent_stats else 3
        
    env_stdout_path=local_pop_dir+agent_id+"/env_log.txt"
    env_stack = make_env_stack(num_envs, game_path, base_port, local_pop_dir+agent_id+"/gamelog.txt", opp_fp_and_elo, last_elo(agent_stats), 
        survivor=agent_stats["survivor"], stdout_path=env_stdout_path, level_path=level_path, image_based=agent_stats["image_based"], time_reward=time_reward, env_p=env_p)
    agent_model_path = last_model_path(model_dir, agent_id, agent_stats)
    agent = PPO.load(agent_model_path, env=env_stack)
    print("Loaded model saved at", agent_model_path, flush=True)
    try:
        # Learn
        agent.learn(total_timesteps=num_steps)
        # Save and cleanup
    except ConnectionError as e:
        env_stack.env_method("kill_env")
        raise e
    except ConnectionResetError as e2:
        env_stack.env_method("kill_env")
        raise e2
    except EOFError as e3:
        env_stack.env_method("kill_env")
        raise e3
    finally:
        env_stack.close()
        del env_stack
    agent_stats["num_steps"] += num_steps
    agent_base = model_dir+agent_id+'/'
    new_agent_save_path = agent_base+agent_id+'_'+str(agent_stats["num_steps"])
    os.system("zip " + agent_base+"archive.zip " + agent_base+"*_*.zip")
    os.system("rm " + agent_base+"*_*.zip")
    agent.save(new_agent_save_path)
    print("Saved model at", new_agent_save_path, flush=True)
    save_stats(model_dir, agent_id, agent_stats)
    
def validate_args(args):
    # Load model and intialize environment
    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Base directory input is not a folder")
        
    if not (args.local_pop_dir[-1] == '/' or args.local_pop_dir[-1] == '\\'):
        args.local_pop_dir = args.local_pop_dir + "/"
    if not os.path.isdir(args.local_pop_dir):
        raise FileNotFoundError("Base directory input is not a folder")
        
    pop = load_pop(args.model_dir)
    if "agent_id" in args and not args.agent_id in pop:
        raise ValueError("Given agent ID is not in population")
    pop = load_pop(args.local_pop_dir)
    if len(pop) <= 0:
        raise ValueError("Population in given base directory does not have any IDs")
    for p in pop:
        if not os.path.isdir(args.model_dir + p):
            raise FileNotFoundError("ID " + p + " does not have a folder in base directory")
        if not os.path.exists(args.model_dir + p + "/stats.json"):
            raise FileNotFoundError("ID " + p + " does not have a stats file")
            
    if args.level_path:
        if not os.path.exists(args.level_path):
            raise FileNotFoundError("Inputted level path", args.level_path, "does not lead to a file")
    
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("local_pop_dir", type=str, help="Base directory for agent models (saved on local host)")
    parser.add_argument("agent_id", type=str, help="ID of agent model to be trained")
    parser.add_argument("--num_steps", type=int, default=100000, help = "Total number of steps to train for")
    parser.add_argument("--base_port", type=int, default=50000, help = "Port that environment will communicate on")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--time_reward", type=float, default=-0.003, help="Reward (or penalty) to give agent at each timestep")
    args = parser.parse_args()
    print(args, flush=True)
    validate_args(args)
    print("Starting training of", args.agent_id, "for", args.num_steps, "steps", flush=True)
    train_agent(args.model_dir, args.local_pop_dir, args.agent_id, args.game_path, args.base_port, args.num_envs, args.num_steps, level_path=args.level_path, time_reward=args.time_reward)
    print("Training of", args.agent_id, "complete", flush=True)