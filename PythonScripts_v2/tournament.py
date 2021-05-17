import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import sys
from train_pop import subset_pop
import eval
import train

def tournament(model_dir, game_path, base_port, num_envs, num_trials, worker_idx, total_workers, reuse_ports=True, level_path=False):
    org_stdout = sys.stdout
    org_stderr = sys.stderr
    my_pop = subset_pop(train.load_pop(model_dir), worker_idx, total_workers)
    results = []
    for i,p in enumerate(my_pop):
        print("Worker", worker_idx, "is starting evaluation of", p, "for", num_trials, "trials per competitor", flush=True)
        sys.stdout = open(model_dir+p+"/tourn_log.txt", 'a')
        sys.stderr = sys.stdout
        p_base_port = base_port if reuse_ports else base_port+(num_envs*i*2)
        j = 0
        last_error = None
        while p_base_port+(j*num_envs*2) < 60000:
            try:
                p_results = eval.evaluate_agent(model_dir, p, game_path, p_base_port+(j*num_envs*2), num_envs, num_trials, level_path=level_path)
                break
            except ConnectionError as e:
                print("ConnectionError detected during tournament, trying a higher port range")
                j += 1
                last_error = e
            except ConnectionResetError as e2:
                print("ConnectionResetError detected during tournament, trying a higher port range")
                j += 1
                last_error = e2
            except EOFError as e3:
                print("EOFError detected during training, trying higher port range")
                j += 1
                last_error = e3
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = org_stdout
        sys.stderr = org_stderr
        if p_base_port+(j*num_envs*2) >= 60000:
            if last_error:
                raise last_error
            else:
                raise ValueError("So there's no last_error, but we got here...?")
        results.append((p,p_results))
        print("Worker", worker_idx, "has completed the evaluation of", p, flush=True)
    return results
    
def record_results(model_dir, agent_id, results):
    agent_stats = train.load_stats(model_dir, agent_id)
    agent_stats["last_eval_steps"] = agent_stats["num_steps"]
    agent_stats["performance"][str(agent_stats["num_steps"])] = results
    train.save_stats(model_dir, agent_id, agent_stats)
        
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
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
    results = tournament(args.model_dir, args.game_path, args.base_port, args.num_envs, args.num_trials, args.worker_idx, args.total_workers, reuse_ports=args.dont_reuse_ports)
    print("Worker", args.worker_idx, "has completed a portion of a tournament", flush=True)
    
    for p,p_results in results:
        record_results(args.model_dir, p, p_results)
        if args.summary:
            eval.print_summary(p, p_results)
    