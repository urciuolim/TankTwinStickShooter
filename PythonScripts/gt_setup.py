import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json

def get_all_model_stats(model_dir):
    all_model_stats = {}
    for subdir, dirs, files in os.walk(model_dir):
        for dir in dirs:
            if "-nemesis" in dir or "-survivor" in dir:
                continue
            if dir[-1] != '/' or dir[-1] != '\\':
                dir += '/'
            with open(subdir + dir + "stats.json", 'r') as stats_file:
                print("Loading", dir, "stats")
                model_stats = json.load(stats_file)
            all_model_stats[str(dir.strip('/'))] = model_stats
    return all_model_stats
    
def get_elo_steps(stats):
    steps = [int(step) for step in stats["elo"].keys()]
    return sorted(steps)
    
def get_starting_steps(stats):
    starting_steps = 0
    parent = stats["parent"]
    while parent:
        starting_steps += int(parent.split('_')[-1])
        parent = all_model_stats[parent.split('_')[0]]["parent"]
    return starting_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Source directory for player base agents")
    parser.add_argument("pb_dir", type=str, help="Directory for player base")
    parser.add_argument("--interval", type=int, default=10000000, help="Interval of number of steps in which agents are selected for player base (i.e. every agent checkpoint that was trained to <interval>*k steps gets brought into the player base)")
    args = parser.parse_args()
    print(args, flush=True)
    
    if args.model_dir[-1] != '/':
        args.model_dir += '/'
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Source directory for agent models is not a folder")
    if args.pb_dir[-1] != '/':
        args.pb_dir += '/'
    if not os.path.isdir(args.pb_dir):
        os.mkdir(args.pb_dir)
    
    all_model_stats = get_all_model_stats(args.model_dir)
    playerbase = []
    for agent in all_model_stats:
        model_stats = all_model_stats[agent]
        model_steps = get_elo_steps(model_stats)
        starting_steps = get_starting_steps(model_stats)
        
        s = -starting_steps
        max_steps = max(model_steps)
        while s <= max_steps:
            if not (s == 0 and model_stats["parent"]):
                if s in model_steps:
                    agent_fp = args.model_dir + agent + '/' + agent + '_' + str(s) + ".zip"
                    if not os.path.exists(agent_fp):
                        raise FileNotFoundError(agent_fp + " should be a checkpoint but is not found")
                    playerbase.append(agent_fp)
            s += args.interval
    
    with open(args.pb_dir + "population.txt", 'w') as playerbase_list:
        for agent_fp in playerbase:
            agent = agent_fp.split('/')[-1].strip(".zip")
            playerbase_list.write(agent + '\n')
            agent_new_dir = args.pb_dir + agent
            os.mkdir(agent_new_dir)
            os.system("cp " + agent_fp + ' ' + agent_new_dir + "/" + agent + "_0.zip")
            starting_stats = {
                "num_steps":0,
                "curr_iter":0,
                "elo": {
                    '0':1000
                },
                "win_rates": {},
                "image_based":all_model_stats[agent.split('_')[0]]["image_based"]
            }
            with open(agent_new_dir + "/stats.json", 'w') as stats_file:
                json.dump(starting_stats, stats_file, indent=4)
    