import argparse
import json
import os

def get_ancestor_steps(agent_id, all_model_stats):
    ancestor_steps = 0
    parent = all_model_stats[agent_id]["parent"]
    while parent:
        ancestor_steps += int(parent.split('_')[-1])
        parent = all_model_stats[parent.split('_')[0]]["parent"]
    return ancestor_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
    parser.add_argument("save_loc", type=str, help="Location to save output file to.")
    args = parser.parse_args()
    print(args)
    
    all_model_stats = {}
    for subdir, dirs, files in os.walk(args.model_dir):
        for dir in dirs:
            if dir[-1] != '/' or dir[-1] != '\\':
                dir += '/'
            with open(subdir + dir + "stats.json", 'r') as stats_file:
                print("Loading", dir, "stats")
                model_stats = json.load(stats_file)
            all_model_stats[str(dir.strip('/'))] = model_stats
           
    all_ancestor_steps = {}
    for agent_id in all_model_stats:
        all_ancestor_steps[agent_id] = get_ancestor_steps(agent_id, all_model_stats)
        
    with open(args.save_loc, 'w') as save_file:
        json.dump(all_ancestor_steps, save_file, indent=4)