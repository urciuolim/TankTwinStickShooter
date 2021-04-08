import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="Base directory for agent models")
parser.add_argument("pop_file_path", type=str, help="Path to file that contains IDs of agents in population to train")
parser.add_argument("N", type=int, help="Number of partitions to consolidate, from 1 to N")
args = parser.parse_args()
print(args)
    
if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
    args.model_dir = args.model_dir + "/"
if not os.path.isdir(args.model_dir):
    raise FileNotFoundError("Base directory for agent models is not a folder")
if not os.path.exists(args.pop_file_path):
    raise FileNotFoundError("Inputted path does not lead to population file")
    
population = []
with open(args.pop_file_path, 'r') as pop_file:
    for line in pop_file.readlines():
        population.append(line.strip('\n'))
      
print("Consolidating training population:", population)

for p in population:
    print("Consolidating", p)
    base_path = args.model_dir + p + '/'
    
    stats_file_path = base_path + "stats.json"
    with open(stats_file_path, 'r') as stats_file:
        model_stats = json.load(stats_file)
    
    if not ("-nemesis" in p or "-survivor" in p):
        # Consolidate performance stats generated from tournament
        avg_reward = model_stats["performance"]["avg_reward"]
        avg_steps = model_stats["performance"]["avg_steps"]
        trained_steps = model_stats["performance"]["trained_steps"]
        
        end = len(trained_steps) - 1
        last_steps = model_stats["num_steps"]
        while end > 0:
            if trained_steps[end-1] != last_steps:
                break
            end -= 1
            
        trained_steps = trained_steps[0:end+1]
        consol_reward = sum(avg_reward[end:])/len(avg_reward[end:])
        avg_reward = avg_reward[0:end]
        avg_reward.append(consol_reward)
        consol_steps = sum(avg_steps[end:])/len(avg_steps[end:])
        avg_steps = avg_steps[0:end]
        avg_steps.append(consol_steps)
        
        model_stats["performance"]["avg_reward"] = avg_reward
        model_stats["performance"]["avg_steps"] = avg_steps
        model_stats["performance"]["trained_steps"] = trained_steps
    
    # Consolidate ELO changes and record them at current num_steps
    elo_change = 0
    for i in range(1, args.N+1):
        with open(base_path + "elo_change-worker" + str(i) + ".txt") as change_file:
            elo_change += int(change_file.readline().strip('\n'))
    model_stats["elo"]["value"].append(model_stats["elo"]["value"][-1] + elo_change)
    model_stats["elo"]["steps"].append(model_stats["num_steps"])
    
    # Write consolidated stats file back
    with open(stats_file_path, 'w') as stats_file:
        json.dump(model_stats, stats_file, indent=4)
    
print("Consolidation complete")