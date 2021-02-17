from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import SAC
import argparse
import os
import json

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, help="Base directory for agent models")
parser.add_argument("id", type=str, help="ID of agent model to be trained")
parser.add_argument("--steps", type=int, default=100000, help = "Total number of steps to train for")
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
    
env = IndvTankEnv(TankEnv(agent=-1, opp_buffer_size=len(opponents)))

model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
if os.path.exists(model_file_path + ".zip"):
    print("Model file found at", model_file_path + ".zip")
    model = SAC.load(model_file_path, env=env, verbose=1)
elif model_stats["num_steps"] > 0:
    raise FileNotFoundError("Model file not found, but stats file indicates that one should exist")
else:
    print("Model file not found, creating new one")
    model = SAC('MlpPolicy', env, verbose=1)
    model.save(model_file_path)

# Load opponents
for opp in opponents:
    opp = opp.strip('\n')
    opp_id = "_".join(opp.split('_')[0:-1])
    env.load_opp_policy(args.base_dir + opp_id + "/" + opp)

# Learn
model.learn(total_timesteps=args.steps)
model_stats["num_steps"] += args.steps
# Save the model
model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
model.save(model_file_path)
print("Saved model at:" + model_file_path)
with open(stats_file_path, 'w') as stats_file:
    json.dump(model_stats, stats_file, indent=4)
    print("Saved stats at:" + stats_file_path)    

env.close()