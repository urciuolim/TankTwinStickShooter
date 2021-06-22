import matplotlib.pyplot as plt
import argparse
import json
import os
from random import random
from plot_pop_elo import get_elo_steps, get_elo_values
import numpy as np

def sorted_keys(all_stats):
    return sorted(all_stats.keys(), key=lambda k: -int(all_stats[k]["elo"][str(all_stats[k]["curr_iter"])]))
    
def get_total_steps(agent_id, anc={}):
    total_steps = int(agent_id.split('_')[-1])
    if agent_id.split('_')[0] in anc:
        total_steps += anc[agent_id.split('_')[0]]
    return total_steps
    
def sorted_keys_by_steps(all_stats, anc={}):
    return sorted(all_stats.keys(), key=lambda k: get_total_steps(k, anc=anc))
    
def safe_div(n,d):
    if d == 0:
        return 0.
    return n/d
    
def get_win_rate_color(all_stats, agent_m, agent_n):
    ID_STEPS=0
    WINS=1
    LOSSES=2
    GAMES=3
    
    total_wins = 0
    total_losses = 0
    total_games = 0
    agent_m_stats = all_stats[agent_m]
    for iter in agent_m_stats["win_rates"]:
        for record in agent_m_stats["win_rates"][iter]:
            if agent_n in record[ID_STEPS]:
                total_wins += record[WINS]
                total_losses += record[LOSSES]
                total_games += record[GAMES]
                break
                
    tie_rate = ((total_wins + total_losses) / total_games)
    if total_wins == total_losses:
        intensity = 0.
        return np.array([intensity, 0., 0., tie_rate])
    elif total_wins > total_losses:
        intensity = safe_div(total_wins-total_losses, total_wins+total_losses)
        return np.array([intensity, 0., 0., tie_rate])
    intensity = safe_div(total_losses-total_wins, total_wins+total_losses)
    return np.array([0., 0., intensity, tie_rate])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
    parser.add_argument("--show_legend", action="store_true", help = "Shows legend with IDs of agents in population.")
    parser.add_argument("--anc", type=str, default=None, help = "Ancestor steps file path, which will adjust ordering of agents in 'who beats who' plot")
    parser.add_argument("--pop_size", type=int, default=24, help = "Size of population at any point in time")
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

    fig,ax = plt.subplots(1,3)
    # Plot main agents elos, also avg reward at each # of trained steps
    colors = {}
    for m in sorted_keys(all_model_stats):
        model_stats = all_model_stats[m]
        colors[m] = (random(), random(), random(), 1)
        # Plot elos
        xs = get_elo_steps(model_stats)
        elos = get_elo_values(model_stats)
        
        ax[0].plot(xs, elos, marker='.', linestyle="None", color=colors[m], label=m)  
            
    ax[0].set_title("Agent ELOs")
    ax[0].set_xlabel("Tournament Iterations")
    ax[0].set_ylabel("ELO")
    ax[0].grid(True)
    leg = ax[0].legend(loc="lower left", prop={"size":6})
    if not args.show_legend: 
        leg.remove()

    fig.suptitle("Results of Tournament of population saved at " + args.model_dir)
    
    num_agents = len(list(all_model_stats.keys()))
    agent_comp = np.ones((num_agents, num_agents, 4), dtype=np.float32)
    
    with open(args.anc, 'r') as anc_file:
        anc = json.load(anc_file)
    sorted_models = sorted_keys_by_steps(all_model_stats, anc=anc)
    for i,m in enumerate(sorted_models):
        print(str(i)+':', m)
        for j,n in enumerate(sorted_models):
            if i == j:
                agent_comp[i,j,:] = np.array([0.,0.,0.,1.])
            else:
                agent_comp[i,j,:] = get_win_rate_color(all_model_stats, m, n)
                
    pop_comp = agent_comp.copy()
    p = args.pop_size
    
    for i in range(0,num_agents,p):
        for j in range(0,num_agents,p):
            pop_comp[i:i+p,j:j+p,0] = np.average(agent_comp[i:i+p,j:j+p,0])
            pop_comp[i:i+p,j:j+p,2] = np.average(agent_comp[i:i+p,j:j+p,2])
            pop_comp[i:i+p,j:j+p,3] = np.average(agent_comp[i:i+p,j:j+p,3])
    
    for i,m in enumerate(sorted_models):
        for j,n in enumerate(sorted_models):
            if i <= j:
                continue
            agent_comp[i,j,:] = np.zeros(4)
            pop_comp[i,j,:] = np.zeros(4)
                
    ax[1].set_title("Agent Win Rates (Alpha Channel = Tie Rate)")
    ax[1].set_xlabel("Blue Channel = This Agent Won More")
    ax[1].set_ylabel("Red Channel = This Agent Won More")
    ax[1].imshow(agent_comp, interpolation='none', vmin=0, vmax=1, aspect='equal')
    ax[1].set_xticks(np.arange(-.5, num_agents, args.pop_size))
    ax[1].set_yticks(np.arange(-.5, num_agents, args.pop_size))
    ax[1].set_xticklabels(np.arange(0, (num_agents//args.pop_size)+1, 1))
    ax[1].set_yticklabels(np.arange(0, (num_agents//args.pop_size)+1, 1))
    ax[1].grid(color='black', linestyle='-', linewidth=1)
    ax[1].xaxis.set_tick_params(labeltop='on')
    
    ax[2].set_title("Popuation Win Rates (Alpha Channel = Tie Rate)")
    ax[2].set_xlabel("Blue Channel = This Population Won More")
    ax[2].set_ylabel("Red Channel = This Population Won More")
    ax[2].imshow(pop_comp, interpolation='none', vmin=0, vmax=1, aspect='equal')
    ax[2].set_xticks(np.arange(-.5, num_agents, args.pop_size))
    ax[2].set_yticks(np.arange(-.5, num_agents, args.pop_size))
    ax[2].set_xticklabels(np.arange(0, (num_agents//args.pop_size)+1, 1))
    ax[2].set_yticklabels(np.arange(0, (num_agents//args.pop_size)+1, 1))
    ax[2].grid(color='black', linestyle='-', linewidth=1)
    ax[2].xaxis.set_tick_params(labeltop='on')
    
    fig.tight_layout()
    plt.show()