from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=100000,
                    help = "Total number of steps to train for")
parser.add_argument("--num_intervals", type=int, default = 10,
                    help = "Total number of intervals to split total steps into")
parser.add_argument("--num_trials", type=int, default=20,
                    help = "Number of trials to run during evaluation")
parser.add_argument("--agent", type=int, default=0, help="Agent to play against RL agent")
parser.add_argument("--id", type=str, default="anon", help="ID to add to paths of saved files")
args = parser.parse_args()
print(args)

env = IndvTankEnv(TankEnv(agent=args.agent))
model = SAC('MlpPolicy', env, verbose=1)
avg_reward = []
avg_steps = []
interval = args.total_steps // args.num_intervals
ID = args.id

for i in range(args.num_intervals):
    model.learn(total_timesteps=interval)
    name = ID + "_sac_TankEnv_" + str((i+1)*interval)
    model.save(name)
    
    total_reward = 0
    total_steps = 0
    for j in range(args.num_trials):
        state = env.reset()
        done = False
        while not done:
            action, _state = model.predict(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1
    avg_reward.append(total_reward/args.num_trials)
    avg_steps.append(total_steps/args.num_trials)
    env.reset()
    
env.close()
fig,ax = plt.subplots(1,2)
ax[0].plot(avg_reward)
ax[0].set_title("Average Reward")
ax[1].plot(avg_steps)
ax[1].set_title("Average Steps during Eval")
plt.savefig(ID + "_results_" + name + ".png")