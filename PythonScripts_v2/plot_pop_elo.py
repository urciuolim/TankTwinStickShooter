import matplotlib.pyplot as plt
import argparse
import json
import os
from random import random

def avg(list):
    return sum([x if x != None else 0 for x in list])/len(list)

def smooth(list, smooth_len):
    avg_list = []
    for i in range(len(list)):
        sublist = list[max(0,i-smooth_len+1):i+1]
        avg_list.append(avg(sublist))
    return avg_list
    
def sorted_main_keys(all_stats, by_elo=True):
    main_keys = [key for key in all_stats.keys() if not (all_stats[key]["nemesis"] or all_stats[key]["survivor"])]
    return sorted(main_keys, key=lambda k: -int(all_stats[k]["elo"][str(all_stats[k]["last_elo_change_steps"])]))
    
def sorted_nem_keys(all_stats):
    nem_keys = [key for key in all_stats.keys() if all_stats[key]["nemesis"]]
    return sorted(nem_keys, key=lambda k: -int(all_stats[k]["elo"][str(all_stats[k]["last_elo_change_steps"])]))
    
def sorted_surv_keys(all_stats):
    surv_keys = [key for key in all_stats.keys() if all_stats[key]["survivor"]]
    return sorted(surv_keys, key=lambda k: -int(all_stats[k]["elo"][str(all_stats[k]["last_elo_change_steps"])]))
    
def get_elo_steps(stats):
    steps = [int(step) for step in stats["elo"].keys()]
    return sorted(steps)
    
def get_elo_values(stats):
    steps = get_elo_steps(stats)
    values = []
    for s in steps:
        values.append(stats["elo"][str(s)])
    return values
    
def get_avg_rewards(stats, matching_agent=None):
    steps = get_elo_steps(stats)
    avg_rewards = []
    for s in steps:
        if s == 0:
          avg_rewards.append(None)
          continue
        avg_reward = 0.
        for record in stats["performance"][str(s)]:
            if matching_agent == record[0].split('_')[0]:
                avg_reward = record[4]
                break
            avg_reward += record[4]
        if not matching_agent:
            avg_reward /= len(stats["performance"][str(s)])
        avg_rewards.append(avg_reward)
    return avg_rewards

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
parser.add_argument("smooth_len", type=int, help = "How many previous values to use during smoothing.")
parser.add_argument("--elo_bucket_num", type=int, default=10, help = "Value to use when bucketing ELOs during counting")
parser.add_argument("--show_legend", action="store_true", help = "Shows legend with IDs of agents in population.")
parser.add_argument("--no_dots", action="store_true", help = "Individual datapoints will not be plotted.")
parser.add_argument("--no_avg", action="store_true", help = "Average elo lines will not be plotted.")
args = parser.parse_args()
print(args)

all_model_stats = {}
nems = False
survs = False
for subdir, dirs, files in os.walk(args.model_dir):
    for dir in dirs:
        if "-nemesis" in dir:
            nems = True
        if "-survivor" in dir:
            survs=True
        if dir[-1] != '/' or dir[-1] != '\\':
            dir += '/'
        with open(subdir + dir + "stats.json", 'r') as stats_file:
            print("Loading", dir, "stats")
            model_stats = json.load(stats_file)
        all_model_stats[str(dir.strip('/'))] = model_stats

fig,ax = plt.subplots(2,2)
# Plot main agents elos, also avg reward at each # of trained steps
colors = {}
for m in sorted_main_keys(all_model_stats):
    model_stats = all_model_stats[m]
    colors[m] = (random(), random(), random(), 1)
    # Plot elos
    xs = get_elo_steps(model_stats)
    parent = model_stats["parent"]
    while parent:
        xs = [x + int(parent.split('_')[-1]) for x in xs]
        parent = all_model_stats[parent.split('_')[0]]["parent"]
    elos = get_elo_values(model_stats)
    
    if not args.no_dots:
        ax[0][0].plot(xs, elos, marker='.', linestyle="None", color=colors[m], label=m)
    
    if not args.no_avg:
        smooth_elos = smooth(elos, args.smooth_len)
        if args.no_dots:
            ax[0][0].plot(xs, smooth_elos, color=colors[m], marker='None', linestyle='--', label=m)
        else:
            ax[0][0].plot(xs, smooth_elos, color=colors[m], marker='None', linestyle='--')
    
    # Plot avg reward
    avg_rewards = get_avg_rewards(model_stats)
    if not args.no_dots:
        ax[0][1].plot(xs, avg_rewards, marker='.', linestyle="None", color=colors[m], label=m)
    
    if not args.no_avg:
        smooth_avg_rewards = smooth(avg_rewards, args.smooth_len)
        if args.no_dots:
            ax[0][1].plot(xs, smooth_avg_rewards, color=colors[m], marker='None', linestyle='--', label=m)
        else:
            ax[0][1].plot(xs, smooth_avg_rewards, color=colors[m], marker='None', linestyle='--')
                
        
        
ax[0][0].set_title("Main agent ELOs")
ax[0][0].set_xlabel("Total steps")
ax[0][0].set_ylabel("ELO")
ax[0][0].grid(True)
leg = ax[0][0].legend(loc="lower left", prop={"size":6})
if not args.show_legend: 
    leg.remove()
    
ax[0][1].set_title("Main agent average")
ax[0][1].set_xlabel("Total steps")
ax[0][1].set_ylabel("Average reward")
ax[0][1].grid(True)
leg = ax[0][1].legend(loc="lower left", prop={"size":6})
if not args.show_legend: 
    leg.remove()
    
if nems:
    # Plot nemesis avg reward at each # of training steps
    for n in sorted_nem_keys(all_model_stats):
        model_stats = all_model_stats[n]
        color = colors[n.replace("-nemesis", "")]
        xs = get_elo_steps(model_stats)
        parent = model_stats["parent"]
        while parent:
            xs = [x + int(parent.split('_')[-1]) for x in xs]
            parent = all_model_stats[parent.split('_')[0]]["parent"]
        avg_rewards = get_avg_rewards(model_stats, matching_agent=n.replace("-nemesis", ""))
        if not args.no_dots:
            ax[1][0].plot(xs, avg_rewards, marker='.', linestyle="None", color=color, label=n)
        
        if not args.no_avg:
            smooth_avg_rewards = smooth(avg_rewards, args.smooth_len)
            if args.no_dots:
                ax[1][0].plot(xs, smooth_avg_rewards, color=color, marker='None', linestyle='--', label=n)
            else:
                ax[1][0].plot(xs, smooth_avg_rewards, color=color, marker='None', linestyle='--')

    ax[1][0].set_title("Nemesis agent average reward (against matching main agent)")
    ax[1][0].set_xlabel("Total steps")
    ax[1][0].set_ylabel("Average reward")
    ax[1][0].grid(True)
    leg = ax[1][0].legend(loc="lower left", prop={"size":6})
    if not args.show_legend: 
        leg.remove()
    
if survs:
    # Plot survivor avg reward at each # of training steps
    for s in sorted_surv_keys(all_model_stats):
        model_stats = all_model_stats[s]
        color = colors[s.replace("-survivor", "")]
        xs = get_elo_steps(model_stats)
        parent = model_stats["parent"]
        while parent:
            xs = [x + int(parent.split('_')[-1]) for x in xs]
            parent = all_model_stats[parent.split('_')[0]]["parent"]
        avg_rewards = get_avg_rewards(model_stats, matching_agent=s.replace("-survivor", ""))
        if not args.no_dots:
            ax[1][1].plot(xs, avg_rewards, marker='.', linestyle="None", color=color, label=s)
        
        if not args.no_avg:
            smooth_avg_rewards = smooth(avg_rewards, args.smooth_len)
            if args.no_dots:
                ax[1][1].plot(xs, smooth_avg_rewards, color=color, marker='None', linestyle='--', label=s)
            else:
                ax[1][1].plot(xs, smooth_avg_rewards, color=color, marker='None', linestyle='--')
                
    ax[1][1].set_title("Survivor agent average reward (against matching main agent)")
    ax[1][1].set_xlabel("Total steps")
    ax[1][1].set_ylabel("Average reward")
    ax[1][1].grid(True)
    leg = ax[1][1].legend(loc="lower left", prop={"size":6})
    if not args.show_legend: 
        leg.remove()

if not nems:
    fig.delaxes(ax[1][0])
if not survs:
    fig.delaxes(ax[1][1])

fig.suptitle("Stats of population saved at " + args.model_dir)
plt.show()