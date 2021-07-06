from preamble import HYPERPARAM_RANGES, choose_hyperp, gen_name, save_new_stats_file
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import json
from elo import elo_prob
from stable_baselines3 import PPO
from train import load_pop, last_elo, load_stats, last_model_path
from random import shuffle

def get_pop_elos_agents_stats(model_dir, flags):
    pop = load_pop(model_dir)
    main_pop = []
    elos = []
    agents = []
    stats = []
    for p in pop:
        if "nemesis" in p:
            flags["has_nemesis"] = True
            continue
        elif "survivor" in p:
            flags["has_survivor"] = True
            continue
        main_pop.append(p)
        stat = load_stats(model_dir, p)
        elos.append(last_elo(stat))
        agents.append(last_model_path(model_dir, p, stat))
        stats.append(stat)
    return list(zip(main_pop, elos, agents, stats))
    
def agent_specific_win_rate(agent1_id, agent1_stat, agent2_id):
    ID = 0
    WINS=1
    GAMES=3
    print(agent1_id)
    agent1_perf = agent1_stat["performance"][str(agent1_stat["last_elo_change_steps"])]
    for record in agent1_perf:
        if "nemesis" in record[ID] or "survivor" in record[ID] or not agent2_id in record[ID]:
            continue
        return record[WINS] / record[GAMES]
        
def save_descendent_model(ancestor_name, ancestor_steps, new_name, model_dir, batch_size=None, n_steps=None,
        n_epochs=None, clip_range=None, gamma=None, gae_lambda=None, vf_coef=None,
        ent_coef=None, learning_rate=None, image_based=False):
    ancestory_policy_path = model_dir + ancestor_name + "/" + ancestor_name + "_" + str(ancestor_steps)
    ancestor = PPO.load(ancestory_policy_path)
    if not batch_size:
        batch_size = choose_hyperp("batch_size", HYPERPARAM_RANGES["batch_size"].index(ancestor.batch_size))
    if not n_steps:
        n_steps = max(batch_size, choose_hyperp("n_steps", HYPERPARAM_RANGES["n_steps"].index(ancestor.n_steps)))//ancestor.n_envs
    if not n_epochs:
        n_epochs = choose_hyperp("n_epochs", HYPERPARAM_RANGES["n_epochs"].index(ancestor.n_epochs))
    if not clip_range:
        clip_range = choose_hyperp("clip_range", HYPERPARAM_RANGES["clip_range"].index(ancestor.clip_range(0)))
    if not gamma:
        gamma = choose_hyperp("gamma", HYPERPARAM_RANGES["gamma"].index(ancestor.gamma))
    if not gae_lambda:
        gae_lambda = choose_hyperp("gae_lambda", HYPERPARAM_RANGES["gae_lambda"].index(ancestor.gae_lambda))
    if not vf_coef:
        vf_coef = choose_hyperp("vf_coef", HYPERPARAM_RANGES["vf_coef"].index(ancestor.vf_coef))
    if not ent_coef:
        ent_coef = choose_hyperp("ent_coef", HYPERPARAM_RANGES["ent_coef"].index(ancestor.ent_coef))
    if not learning_rate:
        learning_rate = choose_hyperp("learning_rate", HYPERPARAM_RANGES["learning_rate"].index(ancestor.learning_rate))
        
    feature_extractor = "MlpPolicy"
    if image_based:
        feature_extractor = "CnnPolicy"
    
    descendent = PPO.load(ancestory_policy_path, batch_size=batch_size, n_steps=n_steps, 
                n_epochs=n_epochs, clip_range=clip_range, gamma=gamma, gae_lambda=gae_lambda,
                vf_coef=vf_coef, ent_coef=ent_coef, learning_rate=learning_rate)
    descendent.save(model_dir + new_name + '/' + new_name + "_0")
    return descendent
        
def descendent_agent(ancestor_name, ancestor_steps, new_name, model_dir, starting_elo, batch_size=None, image_based=False, env_p=3):
    new_agent = save_descendent_model(ancestor_name, ancestor_steps, new_name, model_dir, batch_size=batch_size, image_based=image_based)
    save_new_stats_file(args.model_dir + new_name + "/stats.json", ("image_based", image_based), ("parent", ancestor_name+"_"+str(ancestor_steps)), ("env_p", env_p), starting_elo=starting_elo)
    print("Created", new_name, flush=True)
    return new_agent
    
def descendent_nemesis(ancestor_name, ancestor_steps, new_name, matching_agent, model_dir, starting_elo, batch_size=None, image_based=False, env_p=3):
    ancestor_nemesis = ancestor_name+"-nemesis"
    new_nemesis = new_name+"-nemesis"
    new_agent = save_descendent_model(ancestor_nemesis, ancestor_steps, new_nemesis, model_dir, 
        batch_size=matching_agent.batch_size, n_steps=matching_agent.n_steps, n_epochs=matching_agent.n_epochs, clip_range=matching_agent.clip_range(0),
        gamma=matching_agent.gamma, gae_lambda=matching_agent.gae_lambda, vf_coef=matching_agent.vf_coef, ent_coef=matching_agent.ent_coef,
        learning_rate=matching_agent.learning_rate, image_based=image_based)
    save_new_stats_file(args.model_dir + new_nemesis + "/stats.json", ("image_based", image_based), ("parent", ancestor_nemesis+"_"+str(ancestor_steps)), ("matching_agent", new_name), ("nemesis", True), ("env_p", env_p), starting_elo=starting_elo)
    print("Created", new_nemesis, flush=True)
    
def descendent_survivor(ancestor_name, ancestor_steps, new_name, matching_agent, model_dir, starting_elo, batch_size=None, image_based=False, env_p=3):
    ancestor_survivor = ancestor_name+"-survivor"
    new_survivor = new_name+"-survivor"
    new_agent = save_descendent_model(ancestor_survivor, ancestor_steps, new_survivor, model_dir, 
        batch_size=matching_agent.batch_size, n_steps=matching_agent.n_steps, n_epochs=matching_agent.n_epochs, clip_range=matching_agent.clip_range(0),
        gamma=matching_agent.gamma, gae_lambda=matching_agent.gae_lambda, vf_coef=matching_agent.vf_coef, ent_coef=matching_agent.ent_coef,
        learning_rate=matching_agent.learning_rate, image_based=image_based)
    save_new_stats_file(args.model_dir + new_survivor + "/stats.json", ("image_based", image_based), ("parent", ancestor_survivor+"_"+str(ancestor_steps)), ("matching_agent", new_name), ("survivor", True), ("env_p", env_p), starting_elo=starting_elo)
    print("Created", new_survivor, flush=True)
    
def replace_algorithm(pop_elos_agents_stats, flags, model_dir, noun_fp, adj_fp, batch_size=None, win_thresh=.7, min_steps=10000000):
    pairs = [(a,b) for idx,a in enumerate(pop_elos_agents_stats) for b in pop_elos_agents_stats[idx+1:]]
    shuffle(pairs)
    removed = []
    new_pop = []
    for tuple_a, tuple_b in pairs:
        if tuple_a[1] < tuple_b[1]:
            tmp = tuple_a
            tuple_a=tuple_b
            tuple_b=tmp
        a_id, a_elo, a_agent, a_stat = tuple_a
        b_id, b_elo, b_agent, b_stat = tuple_b
        if a_id in removed or b_id in removed or a_stat["num_steps"] < min_steps or b_stat["num_steps"] < min_steps:
            continue
        if elo_prob(a_elo, b_elo) > win_thresh and agent_specific_win_rate(a_id, a_stat, b_id) > win_thresh:
            print("Replacing", (b_id, b_elo), "with", (a_id, a_elo), flush=True)
            removed.append(b_id)
            new_name = gen_name(noun_fp, adj_fp, model_dir)
            img_bsd = a_stat["image_based"] if "image_based" in a_stat else False
            env_p = a_stat["env_p"] if "env_p" in a_stat else 3
            new_agent = descendent_agent(a_id, a_stat["num_steps"], new_name, model_dir, last_elo(a_stat), batch_size=batch_size, image_based=img_bsd, env_p=env_p)
            new_pop.append(new_name)
            if flags["has_nemesis"]:
                removed.append(b_id + "-nemesis")
                descendent_nemesis(a_id, a_stat["num_steps"], new_name, new_agent, model_dir, last_elo(a_stat), image_based=img_bsd, env_p=env_p)
                new_pop.append(new_name+"-nemesis")
            if flags["has_survivor"]:
                removed.append(b_id + "-survivor")
                descendent_survivor(a_id, a_stat["num_steps"], new_name, new_agent, model_dir, last_elo(a_stat), image_based=img_bsd, env_p=env_p)
                new_pop.append(new_name+"-survivor")
    if len(new_pop) <= 0:
        print("No viable replacements at this time", flush=True)
        return
    with open(model_dir+"population.txt", 'w') as pop_file:
        for p,_,_,_ in pop_elos_agents_stats:
            if not p in removed:
                pop_file.write(p + '\n')
                if flags["has_nemesis"]:
                    pop_file.write(p+'-nemesis\n')
                if flags["has_survivor"]:
                    pop_file.write(p+'-survivor\n')
        for p in new_pop:
            pop_file.write(p + '\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Base directory for agent models")
    parser.add_argument("noun_file_path", type=str, help="Path to noun file used to generate names")
    parser.add_argument("adj_file_path", type=str, help="Path to adj file used to generate names")
    parser.add_argument("--batch_size", type=int, default=None, help="Indicates same batch size for all agents")
    parser.add_argument("--win_thresh", type=float, default=.7, help="Threshold for winning probability that dictates replacements of one agent over another")
    parser.add_argument("--min_steps", type=int, default=10000000, help="Mininum number of steps before models can be considered for replacement")
    args = parser.parse_args()
    print(args, flush=True)
    
    # Validate args
    if not (args.model_dir[-1] == '/' or args.model_dir[-1] == '\\'):
        args.model_dir = args.model_dir + "/"
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError("Base directory for agent models is not a folder")
    if not os.path.exists(args.noun_file_path):
        raise FileNotFoundError("Inputted path does not lead to noun file")
    if not os.path.exists(args.adj_file_path):
        raise FileNotFoundError("Inputted path does not lead to adjective file")
        
    flags = {"has_nemesis":False, "has_survivor":False}
    pop_elos_agents_stats = get_pop_elos_agents_stats(args.model_dir, flags)
    
    replace_algorithm(pop_elos_agents_stats, flags, args.model_dir, args.noun_file_path, args.adj_file_path, batch_size=args.batch_size, win_thresh=args.win_thresh, min_steps=args.min_steps)