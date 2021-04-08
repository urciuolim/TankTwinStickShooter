import matplotlib.pyplot as plt
import argparse
import json
import os
from random import random

def avg(list):
    return sum(list)/len(list)

def smooth(list, smooth_len):
    avg_list = []
    for i in range(len(list)):
        sublist = list[max(0,i-smooth_len+1):i+1]
        avg_list.append(avg(sublist))
    return avg_list

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
for subdir, dirs, files in os.walk(args.model_dir):
    for dir in dirs:
        if dir[-1] != '/' or dir[-1] != '\\':
            dir += '/'
        with open(subdir + dir + "stats.json", 'r') as stats_file:
            model_stats = json.load(stats_file)
        all_model_stats[str(dir.strip('/'))] = model_stats

steps_stats = {}
fig,ax = plt.subplots(1,2)
for m in sorted(all_model_stats.keys(), key=lambda k: -all_model_stats[k]["elo"]["value"][-1]):
        model_stats = all_model_stats[m]
        color = (random(), random(), random(), 1)
        xs = model_stats["elo"]["steps"]
        parent = model_stats["parent"]
        while parent:
            xs = [x + int(parent.split('_')[-1]) for x in xs]
            parent = all_model_stats[parent.split('_')[0]]["parent"]
        elos = model_stats["elo"]["value"]
        
        #Begin steps stats gathering
        for x, elo in zip(xs, elos):
            if not x in steps_stats:
                steps_stats[x] = []
            steps_stats[x].append(elo)
        #End steps stats gathering
        
        if not args.no_dots:
            ax[0].plot(xs, elos, marker='.', linestyle="None", color=color, label=m)
        smooth_elos = smooth(elos, args.smooth_len)
        if not args.no_avg:
            if args.no_dots:
                ax[0].plot(xs, smooth_elos, color=color, marker='None', linestyle='--', label=m)
            else:
                ax[0].plot(xs, smooth_elos, color=color, marker='None', linestyle='--')
                
avg_len = [len(steps_stats[key]) for key in steps_stats]
print(avg(avg_len))
avg_elo = [avg(steps_stats[step]) for step in sorted(steps_stats.keys())]
ax[0].plot(sorted(steps_stats.keys()), smooth(avg_elo, args.smooth_len), marker='None', linestyle='-', color='r')

ax[0].set_title("ELOs throughout training")
ax[0].set_xlabel("Total steps")
ax[0].set_ylabel("ELO")
ax[0].grid(True)
leg = ax[0].legend()
if not args.show_legend: 
    leg.remove()
plt.show()