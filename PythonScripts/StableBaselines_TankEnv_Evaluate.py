import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import PPO
import argparse
import json
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, help="Base directory for agent models")
parser.add_argument("id", type=str, help="ID of agent model to be evaluated")
parser.add_argument("--num_trials", type=int, default=100, help = "Total number of trials to evaluate model for")
parser.add_argument("--no_stats", action="store_true", help="Flag indicates not to record stats after evaluation")
parser.add_argument("--port", type=int, default=50000, help = "Port that environment will communicate on.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
parser.add_argument("--part", type=int, default=1, help="For parallel processing, number of partitions population is being split into")
args = parser.parse_args()
print(args)

# Load model and intialize environment
if not (args.base_dir[-1] == '/' or args.base_dir[-1] == '\\'):
    args.base_dir = args.base_dir + "/"

if not os.path.isdir(args.base_dir):
    raise FileNotFoundError("Base directory input is not a folder")
    
if not os.path.isdir(args.base_dir + args.id):
    raise FileNotFoundError("Inputted ID does not lead to a valid model directory")
    
opp_file_path = args.base_dir + args.id + "/opponents.txt"
stats_file_path = args.base_dir + args.id + "/stats.json"
    
if not os.path.exists(opp_file_path):
    raise FileNotFoundError("Opponents text file not found in model's directory")
    
if not os.path.exists(stats_file_path):
    raise FileNotFoundError("Stats json file not found in model's directory")
    
with open(opp_file_path, 'r') as opp_file:
    opponents = opp_file.readlines()
    if len(opponents) <= 0:
        raise ValueError("No opponents listed in opponents.txt")
        
with open(stats_file_path, 'r') as stats_file:
    model_stats = json.load(stats_file)
    
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

model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
if not os.path.exists(model_file_path + ".zip"):
    raise FileNotFoundError("Model file not found, but stats file indicates that one should exist")

model = PPO.load(model_file_path, env=env_stack, verbose=1)
print("Loaded model named", model_file_path)

# Load opponents
for opp in opponents:
    opp = opp.strip('\n')
    opp_id = "_".join(opp.split('_')[0:-1])
    env_stack.env_method("load_opp_policy", args.base_dir + opp_id + "/" + opp)

total_reward = 0
total_steps = 0
states = env_stack.reset()
envs_done = []
i = 0
while i < args.num_trials:
    reset_states = env_stack.env_method("reset", indices = envs_done)
    for state,env_idx in zip(reset_states, envs_done):
        states[env_idx] = state
    envs_done = []
    while len(envs_done) < 1:
        actions, _states = model.predict(states)
        states, rewards, dones, infos = env_stack.step(actions)
        total_reward += sum(rewards)
        total_steps += args.num_envs
        if any(dones):
            for j,done in enumerate(dones):
                if done:
                    i += 1
                    envs_done.append(j)
    if (i+1) % (args.num_trials / 10) == 0:
        print(((i+1)*100)//args.num_trials, "% trials completed", sep="")
avg_reward = total_reward/i
avg_steps = total_steps/i
print("Average Reward:", avg_reward)
print("Average Steps:", avg_steps)

if not args.no_stats:
    PERF = "performance"
    TRAINED_STEPS = "trained_steps"
    AVG_REWARD = "avg_reward"
    AVG_STEPS = "avg_steps"
    if PERF in model_stats:
        model_stats[PERF][TRAINED_STEPS].append(model_stats["num_steps"])
        model_stats[PERF][AVG_REWARD].append(avg_reward)
        model_stats[PERF][AVG_STEPS].append(avg_steps)
    else:
        model_stats[PERF] = {}
        model_stats[PERF][TRAINED_STEPS] = [model_stats["num_steps"]]
        model_stats[PERF][AVG_REWARD] = [avg_reward]
        model_stats[PERF][AVG_STEPS] = [avg_steps]
        
    with open(stats_file_path, 'w') as stats_file:
        json.dump(model_stats, stats_file, indent=4)
        print("Saved stats at:" + stats_file_path)   

env_stack.close()
with open(args.base_dir + args.id + "/done.txt", 'w') as done_file:
    done_file.write("done")
    print("Wrote done file to", done_file, '\n', args.base_dir + args.id + "/done.txt")