import json
import argparse

def config_gen(config_path, random_start = False):
    with open(config_path, 'r') as openfile:
        json_object = json.load(openfile)
    json_object["player_randomStart"] = random_start

    with open(config_path, 'w') as outfile:
        json.dump(json_object, outfile, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to game config file")
    parser.add_argument("--rs", action="store_true",
                        help="Flag that indicates for players to have random start locations.")
    args = parser.parse_args()
    print("config_gen args:", args)
    
    config_gen(args.config_file, args.rs)