import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import argparse
import json

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, help="Base directory for agent models")
parser.add_argument("id", type=str, help="ID of agent model to be trained")
parser.add_argument("--steps", type=int, default=100000, help = "Total number of steps to train for")
parser.add_argument("--elo", type=int, default=1000, help = "ELO for training agent")
parser.add_argument("--port", type=int, default=50000, help = "Port that environment will communicate on")
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
    
if os.path.exists(stats_file_path):
    print("Stats file found at", stats_file_path)
    with open(stats_file_path, 'r') as stats_file:
        model_stats = json.load(stats_file)
else:
    print("Stats file not found, starting from scratch...")
    model_stats = {"num_steps":0}
    with open(stats_file_path, 'w') as stats_file:
        json.dump(model_stats, stats_file, indent=4)

with open(opp_file_path, 'r') as opp_file:
    opponents = opp_file.readlines()
    if len(opponents) <= 0:
        raise ValueError("No opponents listed in opponents.txt")

envs = []
try:
    for i in range(args.num_envs):
        envs.append(
            lambda a=len(opponents), b=args.elo, c=args.port+(i*args.part), d=model_stats["survivor"]: 
                    IndvTankEnv(TankEnv(agent=-1,
                                        opp_buffer_size=a,
                                        center_elo=b,
                                        game_port=c,
                                        survivor=d
                    ))
        )
    env_stack = SubprocVecEnv(envs, start_method="fork")#[lambda: env for env in envs])
except ConnectionError as e:
    print(e)
    os._exit(2)

model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
if os.path.exists(model_file_path + ".zip"):
    print("Model file found at", model_file_path + ".zip")
    model = PPO.load(model_file_path, verbose=1)
elif model_stats["num_steps"] > 0:
    raise FileNotFoundError("Model file not found, but stats file indicates that one should exist")
else:
    print("Model file not found, creating new one")
    model = PPO('MlpPolicy', env_stack, verbose=1)
    model.save(model_file_path)

# Load opponents
for opp in opponents:
    opp = opp.strip('\n')
    opp_elo = int(opp.split('\t')[-1]) if len(opp.split('\t')) > 1 else args.elo
    opp = opp.split('\t')[0]
    opp_id = "_".join(opp.split('_')[0:-1])
    env_stack.env_method("load_opp_policy", args.base_dir + opp_id + "/" + opp, elo=opp_elo)
    #for env in envs:
    #    env.load_opp_policy(args.base_dir + opp_id + "/" + opp, elo=opp_elo)

# Learn
try:
    model.set_env(env_stack)
    env_stack.reset()
    model.learn(total_timesteps=args.steps)
except TypeError as e:
    print(e)
    print("Despite this error, I'm saving the old policy as the newly trained one")
    model = PPO.load(model_file_path, env=env_stack, verbose=1)
    model_stats["NaN"] = True
except Exception as e:
    print(e)
    #raise(e)
    os._exit(1)
model_stats["num_steps"] += args.steps
# Save the model
model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
model.save(model_file_path)
print("Saved model at:" + model_file_path)
with open(stats_file_path, 'w') as stats_file:
    json.dump(model_stats, stats_file, indent=4)
    print("Saved stats at:" + stats_file_path)    

env_stack.env_method("close")