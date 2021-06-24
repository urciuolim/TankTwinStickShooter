import argparse
import eval
import train
from tournament import tournament
    
def record_results(model_dir, agent_id, results):
    agent_stats = train.load_stats(model_dir, agent_id)
    agent_stats["win_rates"][str(agent_stats["curr_iter"])] = results
    agent_stats["curr_iter"] += 1
    train.save_stats(model_dir, agent_id, agent_stats)
        
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("local_pop_dir", type=str, help="Base directory for agent models (saved on local host)")
    parser.add_argument("--num_trials", type=int, default=50, help = "Number of steps to train each agent for")
    parser.add_argument("--base_port", type=int, default=52000, help="Base port to run game env on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--worker_idx", type=int, default=1, help="Index of worker (for parallel training)")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of workers (for parallel training)")
    parser.add_argument("--dont_reuse_ports", action="store_false", help="Flag indicates that python-side ports should not be reused (for example, because the socket.SO_REUSEADDR option is not supported by OS)")
    parser.add_argument("--summary", action="store_true", help="Flag indicates that summary of tournament should be printed out to stdout")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    args = parser.parse_args()
    print(args, flush=True)
    train.validate_args(args)
    print("Worker", args.worker_idx, "is starting a portion of a tournament", flush=True)
    results = tournament(args.model_dir, args.local_pop_dir, args.game_path, args.base_port, args.num_envs, args.num_trials, args.worker_idx, args.total_workers, 
        reuse_ports=args.dont_reuse_ports, level_path=args.level_path)
    print("Worker", args.worker_idx, "has completed a portion of a tournament", flush=True)
    
    for p,p_results in results:
        record_results(args.model_dir, p, p_results)
        if args.summary:
            eval.print_summary(p, p_results)
    