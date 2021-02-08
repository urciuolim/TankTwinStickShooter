from TankEnv import TankEnv
from RandomAgent import RandomAgent
from AgentLogger import AgentLogger
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--game_ip", type=str, default="127.0.0.1",
                    help = "IP of game server")
parser.add_argument("--game_port", type=int, default = 50000,
                    help = "Port of game server")
parser.add_argument("--connect_attempts", type=int, default=60,
                    help = "Number of attempts to connect to game")
parser.add_argument("--num_games", type=int, default=3,
                    help = "Number of games to play")
parser.add_argument("--players", type=str, default="1|1",
                    help = "Indication of which player is which, each seperated by a '|'. 0=human, 1=random, etc.")
parser.add_argument("--log", action="store_true",
                    help = "Flag indicates that AI agents should log incoming states and outgoing actions")
parser.add_argument("--log_dir", type=str, default=".",
                    help = "Directory to store log files in")
parser.add_argument("--seed", type=int, default=None, help = "Seed to be used for np.random")
parser.add_argument("--no_shoot", action="store_true",
                    help = "Flag that stops AI from shooting")
parser.add_argument("--shoot", action="store_true",
                    help = "Flag that makes AI always shoot")
'''
parser.add_argument("--no_move", action="store_true",
                    help = "Flag that makes AI stay still (will still aim/shoot though)")
parser.add_argument("--no_aim", action="store_true",
                    help = "Flag that makes AI aim straight (relative to their movement direction")
'''
                    
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    if args.seed:
        np.random.seed(args.seed)
        seeds = np.random.randint(0, 2**32-1, size=(args.num_games,))
        
    agents = []
    for i,player in enumerate(args.players.split('|')):
        agent = None
        if player == '1':
            agent = RandomAgent("RandomAgent" + str(i), (52,), np.array([[-1,-1,-1,-1,0], [1,1,1,1,1]]))
        if args.log and player != '0':
            agent = AgentLogger(agent, args.log_dir)
        agents.append(agent)
        
    env = TankEnv(num_agents=2, game_ip = args.game_ip, game_port=args.game_port, num_connection_attempts=args.connect_attempts)
    
    for i in range(args.num_games):
        if args.seed:
            np.random.seed(seeds[i])
            
        state = env.reset()
        start = time.time()
        done = False
        
        while not done:
            actions = np.zeros((2, 5), dtype=np.float32)
            for a,agent in enumerate(agents):
                if agent == None:
                    continue
                action = agent.get_action(state)
                if args.no_shoot:
                    action[-1] = 0
                if args.shoot:
                    action[-1] = 1
                actions[a,:] = action
            state, reward, done, info = env.step(actions)
            
        print("Finished game", i)
        end = time.time()
        print("It took", end-start, "seconds")
        
    env.close()