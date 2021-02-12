from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=100000,
                    help = "Total number of steps to train for")
parser.add_argument("--num_intervals", type=int, default = 10,
                    help = "Total number of intervals to split total steps into")
parser.add_argument("--num_trials", type=int, default=20,
                    help = "Number of trials to run during evaluation")
parser.add_argument("--selfp_model_update", type=int, default=10000,
                    help = "Number of trials to run during evaluation")
parser.add_argument("--id", type=str, default="anon", help="ID to add to paths of saved files")
parser.add_argument("--opp_buf_size", type=int, default=1, help="Size of opponent buffer used in self-play")
args = parser.parse_args()
print(args)

env = IndvTankEnv(TankEnv(agent=-1, old_policy_buffer_size=args.opp_buf_size))
model = SAC('MlpPolicy', env, verbose=1)
avg_reward = []
avg_steps = []
interval = args.total_steps // args.num_intervals
num_selfp_updates = args.total_steps // args.selfp_model_update
ID = args.id
oldname = ID + "_old_policy"
numSteps = 0
lastNumSteps = 0

for i in range(num_selfp_updates):
    # Save over previous old policy (for self-play)
    print("Saving old policy at:", oldname)
    model.save(oldname)
    env.load_old_policy(oldname)
    # Learn
    model.learn(total_timesteps=args.selfp_model_update)
    # Checkpoint policy if at specified interval
    numSteps += args.selfp_model_update
    if (numSteps - lastNumSteps) >= interval:
        name = ID + "_sac_TankEnv_" + str(numSteps)
        model.save(name)
        print("Checkpointed policy named:" + name)
        lastNumSteps = numSteps
        # Also eval policy after checkpoint
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
        print("==========================================")
        print("| Average Reward:", avg_reward[-5:])
        print("| Average Steps:", avg_steps[-5:])
        print("==========================================")
    gc.collect()
    env.reset()
    
env.close()
fig,ax = plt.subplots(1,2)
ax[0].plot(avg_reward)
ax[0].set_title("Average Reward")
ax[1].plot(avg_steps)
ax[1].set_title("Average Steps during Eval")
plt.savefig(ID + "_results_" + name + ".png")