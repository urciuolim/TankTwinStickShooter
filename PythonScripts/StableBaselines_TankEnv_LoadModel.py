from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import SAC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str, help="Path to model to be used")
parser.add_argument("--opp_model_file", type=str, default=None, help="Path to model to be used as opponent")
parser.add_argument("--agent", type=int, default=0, help="Agent to play against RL agent")
parser.add_argument("--rs", action="store_true", help="Random start for tank env")
args = parser.parse_args()
print(args)

env = IndvTankEnv(TankEnv(agent=args.agent, random_side=args.rs))
if args.agent == -1:
    if not args.opp_model_file:
        print("Need to specify an opponent model file when setting agent = -1")
        exit()
    env.load_old_policy(args.opp_model_file)

model = SAC.load(args.model_file)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    if dones:
        obs = env.reset()