import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json
import train
from elo import elo_change

def get_idx_in_results(ID, results):
    ID_STEPS=0
    for i in range(len(results)):
        if ID == results[i][ID_STEPS].split('_')[0]:
            return i
            
def get_last_perf_results(stats):
    return stats["performance"][str(stats["last_eval_steps"])]
    
def recalc_avg_reward(record, is_survivor):
    WINS=1
    LOSSES=2
    GAMES=3
    AVG_REWARD = 4
    AVG_STEPS = 5
    if is_survivor:
        record[AVG_REWARD] = record[AVG_STEPS] / 300
    else:
        record[AVG_REWARD] = (record[WINS] - record[LOSSES])/record[GAMES]
            
def combine_records(record1, record2, survivor_1=False, survivor_2=False):
    WINS=1
    LOSSES=2
    GAMES=3
    AVG_STEPS=5
    
    record1[AVG_STEPS] = (record1[AVG_STEPS]*record1[GAMES] + record2[AVG_STEPS]*record2[GAMES])/(record1[GAMES]+record2[GAMES])
    record2[AVG_STEPS] = record1[AVG_STEPS]
    
    record1_wins = record1[WINS]
    record2_wins = record2[WINS]
    
    record1[WINS] += record2[LOSSES]
    record2[WINS] += record1[LOSSES]
    
    record1[LOSSES] += record2_wins
    record2[LOSSES] += record1_wins
    
    tmp = record1[GAMES]
    record1[GAMES] += record2[GAMES]
    record2[GAMES] += tmp
    
    recalc_avg_reward(record1, survivor_1)
    recalc_avg_reward(record2, survivor_2)

def consolidate_results(pop, all_stats):
    for i in range(len(pop)):
        #print("Consolidating", pop[i])
        i_results = get_last_perf_results(all_stats[i])
        for j in range(i+1, len(pop)):
            j_results = get_last_perf_results(all_stats[j])
            x = get_idx_in_results(pop[j], i_results)
            y = get_idx_in_results(pop[i], j_results)
            combine_records(i_results[x], j_results[y], survivor_1=all_stats[i]["survivor"], survivor_2=all_stats[j]["survivor"])
            
def last_elo(stats):
    return stats["elo"][str(stats["last_elo_change_steps"])]
            
def make_elo_changes(pop, all_stats):
    elo_changes = [0 for _ in pop]
    # Calculate ELO changes for non-exploiters
    for i in range(len(pop)):
        #print("Calculating elo changes for ", pop[i])
        if all_stats[i]["nemesis"] or all_stats[i]["survivor"]:
            continue
        i_results = get_last_perf_results(all_stats[i])
        for (Id_steps, wins, losses, games, avg_reward, avg_steps) in i_results:
            if "-nemesis" in Id_steps or "-survivor" in Id_steps:
                continue
            other_elo = last_elo(all_stats[pop.index(Id_steps.split('_')[0])])
            K=32
            (change, _) = elo_change(last_elo(all_stats[i]), other_elo, K, (avg_reward+1)/2)
            elo_changes[i] += change
            
    print("ELO Changes:", elo_changes)
            
    # Apply ELO changes, using matching agent ELO changes for exploiters
    for i in range(len(pop)):
        i_stats = all_stats[i]
        #print("Making elo changes for ", pop[i])
        if i_stats["nemesis"] or i_stats["survivor"]:
            idx = pop.index(i_stats["matching_agent"])
            i_stats["elo"][str(i_stats["last_eval_steps"])] = last_elo(i_stats) + elo_changes[idx]
        else:
            #print(i_stats)
            i_stats["elo"][str(i_stats["last_eval_steps"])] = last_elo(i_stats) + elo_changes[i]
        i_stats["last_elo_change_steps"] = i_stats["num_steps"]

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    args = parser.parse_args()
    print(args, flush=True)

    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Base directory for agent models is not a folder")
    
    pop = train.load_pop(args.model_dir)
    print("Consolidating training population:", pop, flush=True)
    all_stats = []
    for p in pop:
        all_stats.append(train.load_stats(args.model_dir, p))
    consolidate_results(pop, all_stats)
    make_elo_changes(pop, all_stats)
    for p,s in zip(pop, all_stats):
        train.save_stats(args.model_dir, p, s)