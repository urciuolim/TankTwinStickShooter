import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("stats_file_path", type=str, help = "Path to stats file that holds data to plot.")
parser.add_argument("smooth_len", type=int, help = "How many previous values to use during smoothing.")
args = parser.parse_args()
print(args)

with open(args.stats_file_path, 'r') as stats_file:
    model_stats = json.load(stats_file)

xs = model_stats["performance"]["trained_steps"]
avg_reward = model_stats["performance"]["avg_reward"]
smooth_avg_reward = [None for _ in range(args.smooth_len)]
zero_reward_line = [0 for _ in range(len(avg_reward))]
for i in range(args.smooth_len, len(avg_reward)):
    smooth_avg_reward.append(sum(avg_reward[i-args.smooth_len:i])/args.smooth_len)
avg_steps = model_stats["performance"]["avg_steps"]
smooth_avg_steps = [None for _ in range(args.smooth_len)]
for i in range(args.smooth_len, len(avg_steps)):
    smooth_avg_steps.append(sum(avg_steps[i-args.smooth_len:i])/args.smooth_len)

fig,ax = plt.subplots(1,2)
ax[0].plot(xs, zero_reward_line, color='r')
ax[0].plot(xs, avg_reward)
ax[0].plot(xs, smooth_avg_reward)
ax[0].set_title("Average Reward")
ax[1].plot(xs, avg_steps)
ax[1].plot(xs, smooth_avg_steps)
ax[1].set_title("Average Steps during Eval")
plt.show()