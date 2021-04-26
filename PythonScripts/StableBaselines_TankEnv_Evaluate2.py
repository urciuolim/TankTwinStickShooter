import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import PPO
import argparse
import json
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import subprocess

def evaluate(base_dir, agent_id, env_stack, num_envs, num_trials, no_stats=False):
    stats_file_path = base_dir + agent_id + "/stats.json"
    if not os.path.exists(stats_file_path):
        raise FileNotFoundError("Stats json file not found in model's directory")
    with open(stats_file_path, 'r') as stats_file:
        agent_stats = json.load(stats_file)
    policy_file_path = base_dir + agent_id + "/" + agent_id + "_" + str(agent_stats["num_steps"])
    agent = PPO.load(policy_file_path, env=env_stack, verbose=1)
    print("Loaded policy named", policy_file_path, flush=True)

    total_reward = 0
    total_steps = 0
    states = env_stack.reset()
    envs_done = []
    i = 0
    while i < num_trials:
        reset_states = env_stack.env_method("reset", indices = envs_done)
        for state,env_idx in zip(reset_states, envs_done):
            states[env_idx] = state
        envs_done = []
        while len(envs_done) < 1:
            actions, _states = agent.predict(states)
            states, rewards, dones, infos = env_stack.step(actions)
            total_reward += sum(rewards)
            total_steps += num_envs
            if any(dones):
                for j,done in enumerate(dones):
                    if i >= num_trials:
                        break
                    elif done:
                        i += 1
                        envs_done.append(j)
        if i % (num_trials / 10) == 0:
            print((i*100)//num_trials, "% trials completed", sep="", flush=True)
    avg_reward = total_reward/num_trials
    avg_steps = total_steps/num_trials
    print("Average Reward:", avg_reward, flush=True)
    print("Average Steps:", avg_steps, flush=True)

    if not no_stats:
        PERF = "performance"
        TRAINED_STEPS = "trained_steps"
        AVG_REWARD = "avg_reward"
        AVG_STEPS = "avg_steps"
        if PERF in agent_stats:
            agent_stats[PERF][TRAINED_STEPS].append(agent_stats["num_steps"])
            agent_stats[PERF][AVG_REWARD].append(avg_reward)
            agent_stats[PERF][AVG_STEPS].append(avg_steps)
        else:
            agent_stats[PERF] = {}
            agent_stats[PERF][TRAINED_STEPS] = [agent_stats["num_steps"]]
            agent_stats[PERF][AVG_REWARD] = [avg_reward]
            agent_stats[PERF][AVG_STEPS] = [avg_steps]
            
        with open(stats_file_path, 'w') as stats_file:
            json.dump(agent_stats, stats_file, indent=4)
            print("Saved stats at:" + stats_file_path, flush=True)

    return avg_reward, avg_steps

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("game_config_file_path", type=str, help="File path of game config file")
    parser.add_argument("base_dir", type=str, help="Base directory for agent models")
    parser.add_argument("id", type=str, help="ID of agent model to be evaluated")
    parser.add_argument("--num_trials", type=int, default=100, help = "Total number of trials to evaluate model for")
    parser.add_argument("--no_stats", action="store_true", help="Flag indicates not to record stats after evaluation")
    parser.add_argument("--port", type=int, default=50000, help = "Port that environment will communicate on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--part", type=int, default=1, help="For parallel processing, number of partitions population is being split into")
    parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Text file to direct game logging to")
    args = parser.parse_args()
    print(args, flush=True)

    if not os.path.exists(args.game_path):
        raise FileNotFoundError("Inputted game path does not lead to an existing file")
        
    if not os.path.exists(args.game_config_file_path):
        raise FileNotFoundError("Game config file not found")

    if not (args.base_dir[-1] == '/' or args.base_dir[-1] == '\\'):
        args.base_dir = args.base_dir + "/"

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError("Base directory input is not a folder")
        
    if not os.path.isdir(args.base_dir + args.id):
        raise FileNotFoundError("Inputted ID does not lead to a valid model directory")
        
    opp_file_path = args.base_dir + args.id + "/opponents.txt"
        
    if not os.path.exists(opp_file_path):
        raise FileNotFoundError("Opponents text file not found in model's directory")
        
    with open(opp_file_path, 'r') as opp_file:
        opponents = opp_file.readlines()
        if len(opponents) <= 0:
            raise ValueError("No opponents listed in opponents.txt")
            
    # Run game
    with open(os.path.expanduser(args.gamelog), 'w') as gl:
        game_ps = []
        for i in range(args.num_envs):
            game_cmd_list = [args.game_path, str(args.port+(i*args.part))]
            #config_gen(args.game_config_file_path, random_start=args.rs, port=(50000+args.idx)+(i*args.part))
            game_p = subprocess.Popen(game_cmd_list, stdout=gl, stderr=gl)
            game_ps.append(game_p)
        
    envs = []
    for i in range(args.num_envs):
        envs.append(
            lambda a=len(opponents), c=args.port+(i*args.part): 
                    IndvTankEnv(TankEnv(agent=-1,
                                        opp_buffer_size=a,
                                        game_port=c
                    ))
        )
    env_stack = SubprocVecEnv(envs, start_method="fork")#[lambda: env for env in envs])
    
    # Load opponents
    for opp in opponents:
        opp = opp.strip('\n')
        opp_id = "_".join(opp.split('_')[0:-1])
        env_stack.env_method("load_opp_policy", args.base_dir + opp_id + "/" + opp)
    
    evaluate(args.base_dir, args.id, env_stack, args.num_envs, args.num_trials, no_stats=args.no_stats)

    env_stack.close()
    
    for game_p in game_ps:
        game_p.wait()
        
    with open(args.base_dir + args.id + "/done.txt", 'w') as done_file:
        done_file.write("done")
        print("Wrote done file to", done_file, flush=True)