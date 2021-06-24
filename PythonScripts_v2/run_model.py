from tank_env import TankEnv
from stable_baselines3 import PPO
import numpy as np
import argparse
import matplotlib.pyplot as plt

def run_model(args):
    env = TankEnv(args.game_path, 
        opp_fp_and_elo=[(args.opp, 1000)], 
        game_port=args.base_port, 
        my_port=args.my_port, 
        image_based=args.image_based,
        level_path=args.level_path,
        rand_opp=args.rand_opp)
    model = None
    if args.p1:
        model = PPO.load(args.p1)
    elif args.p1same:
        model = PPO.load(args.opp)
        
    score = [0,0,0]
    print("Score: [Player1 Wins, Player2 Wins, Ties]")
        
    obs = env.reset()
    if args.image_based and (args.ai_view or args.rev_ai_view):
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
    while True:
        if args.image_based and (args.ai_view or args.rev_ai_view):
            if not args.rev_ai_view:
                plt.imshow(obs, origin="lower")
            else:
                plt.imshow(env.opp_state, origin="lower")
            fig.canvas.draw()
        if model:
            action, _ = model.predict(obs)
        elif args.rand_p1:
            action = np.random.rand(5) * 2 - 1
        else:
            action = np.zeros(5, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        if done:
            score[info["winner"]] += 1
            print("Score:", score)
            obs = env.reset()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("opp", type=str, help="Path to model to be used as opponent")
    parser.add_argument("--p1", type=str, default=None, help="Path to model for player 1")
    parser.add_argument("--p1same", action="store_true", help="Indicates that opp model should be used as p1")
    parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
    parser.add_argument("--my_port", type=int, default=50500, help="Port to be used on Python side of network socket connection")
    parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--ai_view", action="store_true", help="Indicates that AI version of game state should be rendered")
    parser.add_argument("--rev_ai_view", action="store_true", help="Indicates that AI version of game state should be rendered, from the perspective of the opponent (red/blue switched)")
    parser.add_argument("--rand_opp", action="store_true", help="Indicates that opponent should be random")
    parser.add_argument("--rand_p1", action="store_true", help="Indicates that player should be random")
    parser.add_argument("--game_path", type=str, default=None, help="File path of game executable")
    args = parser.parse_args()
    print(args)
    run_model(args)