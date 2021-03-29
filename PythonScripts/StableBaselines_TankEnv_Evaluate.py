import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import PPO
import argparse
import json

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, help="Base directory for agent models")
parser.add_argument("id", type=str, help="ID of agent model to be evaluated")
parser.add_argument("--num_trials", type=int, default=100, help = "Total number of trials to evaluate model for")
parser.add_argument("--no_stats", action="store_true", help="Flag indicates not to record stats after evaluation")
parser.add_argument("--port", type=int, default=50000, help = "Port that environment will communicate on.")
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
    
try:
    env = IndvTankEnv(TankEnv(agent=-1, opp_buffer_size=len(opponents), random_opp_sel=False, game_port=args.port))
except ConnectionError as e:
    print(e)
    os._exit(2)

model_file_path = args.base_dir + args.id + "/" + args.id + "_" + str(model_stats["num_steps"])
if not os.path.exists(model_file_path + ".zip"):
    raise FileNotFoundError("Model file not found, but stats file indicates that one should exist")

model = PPO.load(model_file_path, env=env, verbose=1)
print("Loaded model named", model_file_path)

# Load opponents
for opp in opponents:
    opp = opp.strip('\n')
    opp_id = "_".join(opp.split('_')[0:-1])
    env.load_opp_policy(args.base_dir + opp_id + "/" + opp)

total_reward = 0
total_steps = 0
for i in range(args.num_trials):
    state = env.reset()
    done = False
    while not done:
        action, _state = model.predict(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_steps += 1
    if (i+1) % (args.num_trials / 10) == 0:
        print(((i+1)*100)//args.num_trials, "% trials completed", sep="")
avg_reward = total_reward/args.num_trials
avg_steps = total_steps/args.num_trials
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

env.close()