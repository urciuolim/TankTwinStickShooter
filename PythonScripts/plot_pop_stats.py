import matplotlib.pyplot as plt
import argparse
import json
import os
from random import random

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
parser.add_argument("smooth_len", type=int, help = "How many previous values to use during smoothing.")
args = parser.parse_args()
print(args)

fig,ax = plt.subplots(1,3)
for subdir, dirs, files in os.walk(args.model_dir):
    for dir in dirs:
        if dir[-1] != '/' or dir[-1] != '\\':
            dir += '/'
        with open(subdir + dir + "stats.json", 'r') as stats_file:
            model_stats = json.load(stats_file)
        xs = model_stats["performance"]["trained_steps"]
        avg_reward = model_stats["performance"]["avg_reward"]
        #smooth_avg_reward = [None for _ in range(args.smooth_len)]
        #for i in range(args.smooth_len, len(avg_reward)):
        #    smooth_avg_reward.append(sum(avg_reward[i-args.smooth_len:i])/args.smooth_len)
        avg_steps = model_stats["performance"]["avg_steps"]
        #smooth_avg_steps = [None for _ in range(args.smooth_len)]
        #for i in range(args.smooth_len, len(avg_steps)):
        #    smooth_avg_steps.append(sum(avg_steps[i-args.smooth_len:i])/args.smooth_len)
        color = (random(), random(), random(), 1)
        ax[0].plot(xs, avg_reward, marker='.', linestyle="None", color=color, label=dir.split('/')[-2])
        #ax[0].plot(xs, smooth_avg_reward, color=color)
        ax[1].plot(xs, avg_steps, marker='.', linestyle="None", color=color, label=dir.split('/')[-2])
        #ax[1].plot(xs, smooth_avg_steps, color=color)
        xs = model_stats["elo"]["steps"]
        elos = model_stats["elo"]["value"]
        ax[2].plot(xs, elos, marker='.', linestyle="None", color=color, label=dir.split('/')[-2])
        smooth_elos = [None for _ in range(args.smooth_len)]
        for i in range(args.smooth_len, len(elos)):
            smooth_elos.append(sum(elos[i-args.smooth_len:i])/args.smooth_len)
        ax[2].plot(xs, smooth_elos, color=color)

ax[0].set_title("Average Reward")
ax[1].set_title("Average Steps during Eval")
ax[2].set_title("ELOs throughout training")
ax[2].legend()
plt.show()