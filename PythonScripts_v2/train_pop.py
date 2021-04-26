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

def train_multiple_agents(model_dir, game_path, base_port, num_envs, num_steps, worker_idx, total_workers, reuse_ports=True):
    org_stdout = sys.stdout
    org_stderr = sys.stderr
    my_pop = subset_pop(train.load_pop(model_dir), worker_idx, total_workers)
    for i,p in enumerate(my_pop):
        print("Worker", worker_idx, "is starting training of", p, "for", num_steps, "steps", flush=True)
        sys.stdout = open(model_dir+p+"/train_log.txt", 'a')
        sys.stderr = sys.stdout
        p_base_port = base_port if reuse_ports else base_port+(num_envs*i*2)
        train.train_agent(model_dir, p, game_path, p_base_port, num_envs, num_steps)
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = org_stdout
        sys.stderr = org_stderr
        print("Worker", worker_idx, "has completed training of", p, "for", num_steps, "steps", flush=True)
        
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("--num_steps", type=int, default=100000, help = "Number of steps to train each agent for")
    parser.add_argument("--base_port", type=int, default=51000, help="Base port to run game env on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--worker_idx", type=int, default=1, help="Index of worker (for parallel training)")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of workers (for parallel training)")
    parser.add_argument("--dont_reuse_ports", action="store_false", help="Flag indicates that python-side ports should not be reused (for example, because the socket.SO_REUSEADDR option is not supported by OS)")
    args = parser.parse_args()
    print(args, flush=True)
    train.validate_args(args)
    print("Worker", args.worker_idx, "is starting multiple trainings", flush=True)
    train_multiple_agents(args.model_dir, args.game_path, args.base_port, args.num_envs, args.num_steps, args.worker_idx, args.total_workers, reuse_ports=args.dont_reuse_ports)
    print("Worker", args.worker_idx, "has completed multiple trainings", flush=True)