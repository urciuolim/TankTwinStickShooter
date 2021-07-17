import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import train
import sys

def subset_pop(pop, idx, total):
    val = total*(idx-1)
    start = 0 if val == 0 else round((len(pop)/total*(idx-1)))
    end = round((len(pop)/total*(idx)))
    return pop[start:end]

def train_multiple_agents(model_dir, local_pop_dir, game_path, base_port, num_envs, num_steps, worker_idx, total_workers, reuse_ports=True, level_path=None, time_reward=0.):
    org_stdout = sys.stdout
    org_stderr = sys.stderr
    my_pop = subset_pop(train.load_pop(model_dir), worker_idx, total_workers)
    for i,p in enumerate(my_pop):
        print("Worker", worker_idx, "is starting training of", p, "for", num_steps, "steps", flush=True)
        sys.stdout = open(model_dir+p+"/train_log.txt", 'a')
        sys.stderr = sys.stdout
        p_base_port = base_port if reuse_ports else base_port+(num_envs*i*2)
        j = 0
        last_error = None
        while p_base_port+(j*num_envs*2) < 60000:
            try:
                train.train_agent(model_dir, local_pop_dir, p, game_path, p_base_port+(j*num_envs*2), num_envs, num_steps, level_path=level_path, time_reward=time_reward)
                break
            except ConnectionError as e:
                print("ConnectionError detected during training, trying a higher port range")
                j += 1
                last_error = e
            except ConnectionResetError as e2:
                print("ConnectionResetError detected during training, trying a higher port range")
                j += 1
                last_error = e2
            except EOFError as e3:
                print("EOFError detected during training, trying higher port range")
                j += 1
                last_error = e3
            except json.decoder.JSONDecodeError as e4:
                print("JSONDecodeError detected during training, trying higher port range")
                j += 1
                last_error = e4
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = org_stdout
        sys.stderr = org_stderr
        if p_base_port+(j*num_envs*2) >= 60000:
            if last_error:
                raise last_error
            else:
                raise ValueError("So there's no last_error, but we got here...?")
        print("Worker", worker_idx, "has completed training of", p, "for", num_steps, "steps", flush=True)
        
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("local_pop_dir", type=str, help="Base directory for agent models (saved on local host)")
    parser.add_argument("--num_steps", type=int, default=100000, help = "Number of steps to train each agent for")
    parser.add_argument("--base_port", type=int, default=51000, help="Base port to run game env on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--worker_idx", type=int, default=1, help="Index of worker (for parallel training)")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of workers (for parallel training)")
    parser.add_argument("--dont_reuse_ports", action="store_false", help="Flag indicates that python-side ports should not be reused (for example, because the socket.SO_REUSEADDR option is not supported by OS)")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    parser.add_argument("--time_reward", type=float, default=0., help="Reward (or penalty) to give agent at each timestep")
    args = parser.parse_args()
    print(args, flush=True)
    train.validate_args(args)
    print("Worker", args.worker_idx, "is starting multiple trainings", flush=True)
    train_multiple_agents(args.model_dir, args.local_pop_dir, args.game_path, args.base_port, args.num_envs, args.num_steps, args.worker_idx, args.total_workers, 
        reuse_ports=args.dont_reuse_ports, level_path=args.level_path, time_reward=args.time_reward)
    print("Worker", args.worker_idx, "has completed multiple trainings", flush=True)