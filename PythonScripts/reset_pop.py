import argparse
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
parser.add_argument("pop_file_path", type=str, help="Path to file that contains IDs of agents in population to train")
parser.add_argument("num_steps", type=int, help = "Number of steps to reset population to")
args = parser.parse_args()
print(args)

all_model_stats = {}
for subdir, dirs, files in os.walk(args.model_dir):
    for dir in dirs:
        if dir[-1] != '/' or dir[-1] != '\\':
            dir += '/'
        with open(subdir + dir + "stats.json", 'r') as stats_file:
            model_stats = json.load(stats_file)
        all_model_stats[str(dir.strip('/'))] = model_stats
        
population = []
with open(args.pop_file_path, 'r') as pop_file:
    for line in pop_file.readlines():
        population.append(line.strip('\n'))

print("Resetting population to", args.num_steps, "steps")

agents_to_remove = []
for p in population:
    print("Resetting", p)
    base_path = args.model_dir + p + '/'
    
    stats_file_path = base_path + "stats.json"
    with open(stats_file_path, 'r') as stats_file:
        model_stats = json.load(stats_file)
        
    parent = model_stats["parent"]
    ancestor_num_steps = 0
    while parent:
        ancestor_num_steps += int(parent.split('_')[-1])
        parent = all_model_stats[parent.split('_')[0]]["parent"]
    
    new_num_steps = args.num_steps - ancestor_num_steps
    if new_num_steps < 0:
        population.append(model_stats["parent"].split('_')[0])
        agents_to_remove.append(p)
    else:
        model_stats["num_steps"] = new_num_steps
        for i,trained_steps in enumerate(model_stats["performance"]["trained_steps"]):
            if trained_steps > new_num_steps:
                model_stats["performance"]["avg_reward"] = model_stats["performance"]["avg_reward"][:i]
                model_stats["performance"]["avg_steps"] = model_stats["performance"]["avg_steps"][:i]
                model_stats["performance"]["trained_steps"] = model_stats["performance"]["trained_steps"][:i]
                
        for i,steps in enumerate(model_stats["elo"]["steps"]):
            if steps > new_num_steps:
                model_stats["elo"]["value"] = model_stats["elo"]["value"][:i]
                model_stats["elo"]["steps"] = model_stats["elo"]["steps"][:i]
    
        # Write consolidated stats file back
        with open(stats_file_path, 'w') as stats_file:
            json.dump(model_stats, stats_file, indent=4)
            
with open(args.pop_file_path, 'w') as pop_file:
    for p in population:
        pop_file.write(p + '\n')
    
print("Reset complete")