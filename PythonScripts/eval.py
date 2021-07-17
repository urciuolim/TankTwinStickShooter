import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from tank_env import TankEnv
from stable_baselines3 import PPO
import argparse
import json
import train

DUMMY_ELO=1000

def curr_model_path(model_dir, agent_id, agent_stats):
    return model_dir+agent_id+'/'+agent_id+'_'+str(agent_stats["num_steps"])

def get_opps_and_elos(model_dir, agent_id):
    pop = train.load_pop(model_dir)
    pop.remove(agent_id)
    
    pop_fps = []
    for p in pop:
        p_stats = train.load_stats(model_dir, p)
        pop_fps.append(curr_model_path(model_dir, p, p_stats))
    return list(zip(pop_fps, [DUMMY_ELO for _ in pop_fps]))

def evaluate_agent(model_dir, local_pop_dir, agent_id, game_path, base_port, num_envs, num_trials, level_path=None):
    PLAYER_1=0
    PLAYER_2=1
    FP=0
    # Load agent and env
    agent_stats = train.load_stats(model_dir, agent_id)
    #if not (agent_stats["nemesis"] or agent_stats["survivor"]):
    opp_fp_and_elo = get_opps_and_elos(local_pop_dir, agent_id)
    #else:
        #opp_fp_and_elo = [(curr_model_path(model_dir, agent_stats["matching_agent"], train.load_stats(model_dir, agent_stats["matching_agent"])), DUMMY_ELO)]
        
    env_p = agent_stats["env_p"] if "env_p" in agent_stats else 3
        
    env_stdout_path=local_pop_dir+agent_id+"/env_log.txt"
    env_stack = train.make_env_stack(num_envs, game_path, base_port, local_pop_dir+agent_id+"/gamelog.txt", opp_fp_and_elo, DUMMY_ELO, 
        elo_match=False, survivor=agent_stats["survivor"] if "survivor" in agent_stats else False, stdout_path=env_stdout_path, level_path=level_path, image_based=agent_stats["image_based"], env_p=env_p)
    agent_model_path = curr_model_path(model_dir, agent_id, agent_stats)
    agent = PPO.load(agent_model_path, env=env_stack)
    print("Loaded model saved at", agent_model_path, flush=True)
    try:
    # Evaluate
        results = []
        for i,(opp_fp, _) in enumerate(opp_fp_and_elo):
            print("Starting evaluation of", agent_id, "vs", opp_fp, flush=True)
            print(i*100/len(opp_fp_and_elo), "% complete", sep="", flush=True)
            total_reward = 0
            total_steps = 0
            total_wins = 0
            total_losses = 0
            states = env_stack.reset()
            envs_done = []
            running_reward = [0 for _ in range(num_envs)]
            running_steps = [0 for _ in range(num_envs)]
            i = 0
            while i < num_trials:
                reset_states = env_stack.env_method("reset", indices = envs_done)
                for state,env_idx in zip(reset_states, envs_done):
                    states[env_idx] = state
                envs_done = []
                while len(envs_done) < 1:
                    actions, _ = agent.predict(states)
                    states, rewards, dones, infos = env_stack.step(actions)
                    for k in range(num_envs):
                        running_reward[k] += rewards[k]
                        running_steps[k] += 1
                    if any(dones):
                        for j,done in enumerate(dones):
                            if done:
                                i += 1
                                envs_done.append(j)
                                total_reward += running_reward[j]
                                running_reward[j] = 0
                                total_steps += running_steps[j]
                                running_steps[j] = 0
                                if "winner" in infos[j]:
                                    if infos[j]["winner"] == PLAYER_1:
                                        total_wins += 1
                                    elif infos[j]["winner"] == PLAYER_2:
                                        total_losses += 1
            avg_reward = total_reward/i
            avg_steps = total_steps/i
            results.append((opp_fp.split('/')[-1], total_wins, total_losses, i, avg_reward, avg_steps))
            if opp_fp != opp_fp_and_elo[-1][FP]:
                env_stack.env_method("next_opp")
    except ConnectionError as e:
        env_stack.env_method("kill_env")
        raise e
    except ConnectionResetError as e2:
        env_stack.env_method("kill_env")
        raise e2
    except EOFError as e3:
        env_stack.env_method("kill_env")
        raise e3
    except json.decoder.JSONDecodeError as e4:
        env_stack.env_method("kill_env")
        raise e4
    finally:
        # Cleanup and return
        env_stack.close()
        del env_stack
    return results
    
def print_summary(agent_id, results):
    print("Summary of evaluation:", flush=True)
    for (opp_fp, total_wins, total_losses, total_games, avg_reward, avg_steps) in results:
        print("Evaluation results of", agent_id, "vs", opp_fp)
        print("\tTotal Wins:", total_wins)
        print("\tTotal Losses:", total_losses)
        print("\tTotal Ties:", total_games-(total_wins+total_losses))
        print("\tTotal Games:", total_games)
        print("\tAverage Reward:", avg_reward)
        print("\tAverage Steps:", avg_steps, flush=True)
    
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("game_path", type=str, help="File path of game executable")
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("local_pop_dir", type=str, help="Base directory for agent models (saved on local host)")
    parser.add_argument("agent_id", type=str, help="ID of agent model to be evaluated")
    parser.add_argument("--num_trials", type=int, default=50, help = "Total number of trials to evaluate model for")
    parser.add_argument("--base_port", type=int, default=52000, help = "Base port that environments will communicate on.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run concurrently")
    parser.add_argument("--level_path", type=str, default=None, help="Path to level file")
    args = parser.parse_args()
    print(args, flush=True)
    train.validate_args(args)
    print("Starting evaluation of", args.agent_id, "with", args.num_trials, "trials against each opponent in population", flush=True)
    results = evaluate_agent(args.model_dir, args.local_pop_dir, args.agent_id, args.game_path, args.base_port, args.num_envs, args.num_trials, level_path=args.level_path)
    print("Evaluation of", args.agent_id, "complete", flush=True)
    print_summary(results)