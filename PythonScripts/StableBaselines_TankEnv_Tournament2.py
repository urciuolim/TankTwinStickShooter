import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
from config_gen import config_gen
import json
from elo import *
import subprocess
from StableBaselines_TankEnv_Evaluate2 import evaluate
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
import sys

def get_steps(stats_file_path):
    with open(stats_file_path, 'r') as stats_file:
        return json.load(stats_file)["num_steps"]
    
def safe_get_elo(model_stats):
    if not "elo" in model_stats:
        model_stats["elo"] = {}
        model_stats["elo"]["value"] = [1000]
        model_stats["elo"]["steps"] = [model_stats["num_steps"]]
    return model_stats["elo"]["value"][-1]
    
def get_reward(model_stats):
    last_reward = model_stats["performance"]["avg_reward"][-1]
    #model_stats["performance"]["avg_reward"] = model_stats["performance"]["avg_reward"][:-1]
    #model_stats["performance"]["avg_steps"] = model_stats["performance"]["avg_steps"][:-1]
    #model_stats["performance"]["trained_steps"] = model_stats["performance"]["trained_steps"][:-1]
    return (last_reward + 1.) / 2.
    
def apply_elo_change(agent, elo_change):
    agent_stats_file_path = args.model_dir + agent + "/stats.json"
    with open(agent_stats_file_path, 'r') as agent_stats_file:
        agent_stats = json.load(agent_stats_file)
    if not "elo" in agent_stats:
        agent_stats["elo"] = {"value":[1000], "steps":[agent_stats["num_steps"]]}
    agent_stats["elo"]["value"].append(safe_get_elo(agent_stats) + elo_change)
    agent_stats["elo"]["steps"].append(int(agent_stats["num_steps"]))
    with open(agent_stats_file_path, 'w') as agent_stats_file:
        json.dump(agent_stats, agent_stats_file, indent=4)

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, help="File path of game executable")
parser.add_argument("game_config_file_path", type=str, help="File path of game config file")
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("comp_file_path", type=str, help="File listing all competitors for tournament")
parser.add_argument("--num_trials", type=int, default=50, help="Number of trials for each pair of competitors to play out")
parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to direct game logging to")
parser.add_argument("--idx", type=int, default=0, help="For parallel processing, portion of population this job will train (from 1 to {args.part} )")
parser.add_argument("--part", type=int, default=1, help="For parallel processing, number of partitions population is being split into")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
parser.add_argument("--base_port", type=int, default=52000, help="Base port to run game env on.")
args = parser.parse_args()
print(args, flush=True)
    
if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
    args.model_dir = args.model_dir + "/"
    
if not os.path.exists(args.game_path):
    raise FileNotFoundError("Inputted game path does not lead to an existing file")
    
if not os.path.exists(args.game_config_file_path):
    raise FileNotFoundError("Game config file not found")
    
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
    
if not os.path.exists(args.comp_file_path):
    raise FileNotFoundError("Competitors file not found")
    
original_stdout = sys.stdout
    
competitors = []
HAS_NEM = False
HAS_SURV = False
with open(args.comp_file_path, 'r') as comp_file:
    for line in comp_file.readlines():
        if "-nemesis" in line:
            HAS_NEM = True
            continue
        if "-survivor" in line:
            HAS_SURV = True
            continue
        competitors.append(line.strip('\n'))

for c in competitors:
    if not os.path.isdir(args.model_dir + c):
        raise FileNotFoundError("Competitor ID {" + c + "} does not lead to a valid model directory")
    if not os.path.exists(args.model_dir + c + "/stats.json"):
        raise FileNotFoundError("Competitor ID (" + c + ") does not have a stats file")
        
if args.idx and args.part > 1:
    start = round((len(competitors)/args.part*(args.idx-1)))
    end = round((len(competitors)/args.part*(args.idx)))
    my_comps = competitors[start:end]
else:
    my_comps = competitors
        
elo_changes = [0 for _ in range(len(competitors))]

with open(os.path.expanduser(args.gamelog), 'w') as gl:
    game_ps = []
    for i in range(args.num_envs):
        game_cmd_list = [args.game_path, str((args.base_port+args.idx)+(i*args.part))]
        #config_gen(args.game_config_file_path, random_start=args.rs, port=(50000+args.idx)+(i*args.part))
        game_p = subprocess.Popen(game_cmd_list, stdout=gl, stderr=gl)
        game_ps.append(game_p)
        
    envs = []
    for i in range(args.num_envs):
        envs.append(
            lambda c=(args.base_port+args.idx)+(i*args.part): 
                    IndvTankEnv(TankEnv(agent=-1,
                                        opp_buffer_size=1,
                                        game_port=c
                    ))
        )
    env_stack = SubprocVecEnv(envs, start_method="fork")#[lambda: env for env in envs])
    # Each competitor will play each other <args.num_trials> times
    for c in my_comps:
        sys.stdout = open(args.model_dir + c + "/tourn_log.txt", 'a')
        for j,opp in enumerate(competitors):
            if c == opp: # Doesn't make sense to play itself as ELO scores are being adjusted
                continue
            env_stack.env_method("unload_opp_policies")
                
            print("Competitors:", competitors, flush=True)
            print("elo_changes:", elo_changes, flush=True)
            print("My competitors:", my_comps, flush=True)
            print((j/len(competitors))*100, "% complete with tournament", sep="", flush=True)
            c_stats_file_path = args.model_dir + c + "/stats.json"
            opp_stats_file_path = args.model_dir + opp + "/stats.json"
            opp_file_path = args.model_dir + c + "/opponents.txt"
            
            c_steps = get_steps(c_stats_file_path)
            opp_steps = get_steps(opp_stats_file_path)
                    
            print(c+ "_" + str(c_steps), "vs", opp + "_" + str(opp_steps), flush=True)

            # Establish opponents for model to play against
            #with open(opp_file_path, 'w') as opp_file:
                #opp_file.write(opp + "_" + str(opp_steps))
            env_stack.env_method("load_opp_policy", args.model_dir + opp + '/' + opp + '_' + str(opp_steps))
                
            # Execute evaluation
            print("Starting tournament eval:", c + "_" + str(c_steps), "vs", opp + "_" + str(opp_steps), "by worker", args.idx, flush=True)
            (avg_reward, _) = evaluate(args.model_dir, c, env_stack, args.num_envs, args.num_trials, no_stats=True)
            c_win_rate = (avg_reward + 1.) / 2.
            print("Ending tournament eval", flush=True)

            '''
            if os.path.exists(args.model_dir + c + "/done.txt"):
                print("Worker", args.idx, "has completed the tournament eval of", c)
                os.remove(args.model_dir + c + "/done.txt")
            else:
                print("Worker", args.idx, "did not complete the tournament eval of", c, "so ...")
            '''
            
            with open(c_stats_file_path, 'r') as c_stats_file:
                c_stats = json.load(c_stats_file)
                
            with open(opp_stats_file_path, 'r') as opp_stats_file:
                opp_stats = json.load(opp_stats_file)
                
            #c_avg_reward = get_reward(c_stats)
            c_elo = safe_get_elo(c_stats)
            opp_elo = safe_get_elo(opp_stats)
            K = 32
            c_elo_change, opp_elo_change = elo_change(c_elo, opp_elo, K, c_win_rate)
            #NOTE: If ELO seems to continuously inflate during PBT, perhaps implement provisional ELO ratings for new agents?
            elo_changes[competitors.index(c)] += c_elo_change
            elo_changes[j] += opp_elo_change
    # end for c in my_comps:
            
    env_stack.close()
    for game_p in game_ps:
        game_p.wait()
        
sys.stdout.close()
sys.stdout = original_stdout
            
print("elo_changes:", elo_changes, flush=True)
            
exploiters = []
if HAS_NEM:
    exploiters.append("-nemesis")
if HAS_SURV:
    exploiters.append("-survivor")
if args.idx and args.part > 1:
    for elo_change,c in zip(elo_changes, competitors):
        c_change_file_path = args.model_dir + c + "/elo_change-worker" + str(args.idx) + ".txt"
        with open(c_change_file_path, 'w') as c_change_file:
            c_change_file.write(str(elo_change))
        for e in exploiters:
            e_change_file_path = args.model_dir + c + e + "/elo_change-worker" + str(args.idx) + ".txt"
            with open(e_change_file_path, 'w') as e_change_file:
                e_change_file.write(str(elo_change))
else:
    for elo_change,c in zip(elo_changes, competitors):
        apply_elo_change(c, elo_change)
        for e in exploiters:
            apply_elo_change(c+e, elo_change)