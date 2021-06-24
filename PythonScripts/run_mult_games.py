import subprocess
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("init_file_path", type=str, help="Path to init file, which outlines which agents play against what. Each line needs to have two names seperated by a tab, one for each agent (i.e. path/to/model    random")
parser.add_argument("game_path", type=str, default=None, help="File path of game executable")
parser.add_argument("--base_port", type=int, default=50000, help="Base port to be used for game environment")
parser.add_argument("--image_based", action="store_true", help="Indicates that env observation space is image based")
parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
args = parser.parse_args()
print(args)

with open(args.init_file_path, 'r') as init_file:
    inits = init_file.readlines()

try:
    game_ps = []    
    for i,init in enumerate(inits):
        p1, p2 = init.strip('\n').split('\t')
        game_cmd_list = ["python", "run_model.py",
                        "--game_path", args.game_path,
                        "--base_port", str(args.base_port+(2*i)),
                        "--my_port", str(args.base_port+(2*i+1))]
                        
        if p1 == "random":
            game_cmd_list.append("--rand_p1")
        else:
            game_cmd_list.append("--p1")
            game_cmd_list.append(p1)
            
        if p2 == "random":
            game_cmd_list.append(".")
            game_cmd_list.append("--rand_opp")
        else:
            game_cmd_list.append(p2)
            
        if args.image_based:
            game_cmd_list.append("--image_based")
        if args.level_path:
            game_cmd_list.append("--level_path")
            game_cmd_list.append(args.level_path)
        game_ps.append(subprocess.Popen(game_cmd_list))
        
        time.sleep(5)
    for game_p in game_ps:
        game_p.wait()
finally:
    for game_p in game_ps:
        game_p.kill()
    print("Killed all python subprocesses (you may need to manually kill the game processes)")