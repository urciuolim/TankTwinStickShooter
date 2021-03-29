from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3 import PPO
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("game_path", type=str, help="File path of game executable")
parser.add_argument("model_path", type=str, help="Path to saved model")
parser.add_argument("--gamelog", type=str, default="gamelog.txt", help="Log file to redirect game logging to")
args = parser.parse_args()
print(args)

with open(args.gamelog, 'w') as gl:
    print("Starting game")
    game_p = subprocess.Popen(args.game_path, stdout=gl, stderr=gl)
    print("Initializing env")
    env = IndvTankEnv(TankEnv())
    print("Loading model")
    m = PPO.load(args.model_path, env=env, batch_size=61, n_steps=1953)
    print(m.get_parameters())
    print("Setting up learn")
    (_,c) = m._setup_learn(m.n_steps, m.env)
    print("Collecting rollouts")
    m.collect_rollouts(m.env, c, m.rollout_buffer, m.n_steps)
    m.n_epochs = 1
    env.close()
    game_p.kill()
    print("Training model for 1 epoch")
    m.train()
    print("\npolicy_net:")
    print(list(m.policy.mlp_extractor.policy_net.children())[0].weight.grad)
    print(list(m.policy.mlp_extractor.policy_net.children())[2].weight.grad)
    print("\nvalue_net:")
    print(list(m.policy.mlp_extractor.value_net.children())[0].weight.grad)
    print(list(m.policy.mlp_extractor.value_net.children())[2].weight.grad)
    print("\naction_net:")
    print(m.policy.action_net.weight.grad)
    print(m.policy.action_net.bias.grad)
    print("\nvalue_net (output):")
    print(m.policy.value_net.weight.grad)
    print(m.policy.value_net.bias.grad)