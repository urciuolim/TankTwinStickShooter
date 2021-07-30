import gym
from human_matchmaking import opp_fp
from elo import elo_change
from tank_env import choice_with_normalization, weight_func, TankEnv
import numpy as np

def combine_winrates(all_stats):
    ID,WINS,LOSSES,GAMES=(0,1,2,3)
    for p in all_stats:
        p_combined_winrates = {}
        p_winrates = all_stats[p]["win_rates"]
        for iter in p_winrates:
            for record in p_winrates[iter]:
                opp = record[ID]
                if len(opp.split('_')) > 2:
                    opp = "_".join(opp.split('_')[:-1])
                if not opp in p_combined_winrates:
                    p_combined_winrates[opp] = record[:GAMES+1]
                else:
                    for i in range(WINS,GAMES+1):
                        p_combined_winrates[opp][i] += record[i]
        all_stats[p]["win_rates"] = p_combined_winrates
    return all_stats

class AIMatchmaker(gym.Env):
    metadata = {'render.modes': None}

    def __init__(self,
                all_stats, 
                all_opps, 
                all_elos,
                game_path,
                model_dir,
                base_port=50000,
                my_port=50001,
                image_based=False,
                level_path=None,
                env_p=3,
                starting_elo=None,
                K=16,
                D=5.,
                time_reward=-0.003,
                matchmaking_mode=0,
                elo_log_interval=10000,
                win_loss_ratio=[0,0]
    ):
        super(AIMatchmaker, self).__init__()
        
        self.all_stats = combine_winrates(all_stats)
        self.all_opps = all_opps
        self.all_elos = all_elos
        self.model_dir = model_dir
        
        self.agent_elo = starting_elo if starting_elo != None else self.all_elos[0]
        self.env = TankEnv(game_path, 
            opp_fp_and_elo=[], 
            game_port=base_port, 
            my_port=my_port, 
            image_based=image_based,
            level_path=level_path,
            p=env_p,
            time_reward=time_reward
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.K = K
        self.D = D
        self.my_port = my_port
        self.mm = matchmaking_mode
        
        self.uncounted_games = np.array([0,0], dtype=np.uint32)
        self.counted_game_sets = 0
        self.win_loss_ratio = np.array(win_loss_ratio, dtype=np.uint32)
        
        self.started = False
        self.next_opp()
        
        self.elo_log_interval = elo_log_interval
        self.num_steps = 0
        self.elo_log = []
        
    def next_opp(self):
        weights = np.zeros((len(self.all_elos)), dtype=np.float32)
        if self.mm == 1:
            # ELO based matchmaking, where ELOs closer to agent ELo is prefered (but not guarenteed)
            weights += np.array([weight_func(elo-self.agent_elo, self.D) for elo in self.all_elos], dtype=np.float32)
                
        if any(self.win_loss_ratio):
            while all(self.uncounted_games >= self.win_loss_ratio):
                self.uncounted_games -= self.win_loss_ratio
                self.counted_game_sets += 1
                
            tmp = self.uncounted_games >= self.win_loss_ratio
            if tmp[0] and not tmp[1]:
                # Need more losses
                if self.mm == 1:
                    # Zero weights for opponents that have <= ELOs than agent
                    for i,elo in enumerate(self.all_elos):
                        if elo <= self.agent_elo:
                            weights[i] = 0
                    # Choose agent with highest ELO if agent ELO is higher than all opponent ELOs
                    if sum(weights) == 0:
                        weights[self.all_elos.index(max(self.all_elos))] = 1
                else:
                    # Equal probability for opponents that have > ELOs than agent
                    for i,elo in enumerate(self.all_elos):
                        if elo > self.agent_elo:
                            weights[i] = 1
                    # Choose agent with highest ELO if agent ELO is higher than all opponent ELOs
                    if sum(weights) == 0:
                        weights[self.all_elos.index(max(self.all_elos))] = 1
            elif not tmp[0] and tmp[1]:
                # Need more wins
                if self.mm == 1:
                    # Zero weights for opponents that have >= ELOs than agent
                    for i,elo in enumerate(self.all_elos):
                        if elo >= self.agent_elo:
                            weights[i] = 0
                    # Choose agent with lowest ELO if agent ELO is higher than all opponent ELOs
                    if sum(weights) == 0:
                        weights[self.all_elos.index(min(self.all_elos))] = 1
                else:
                    # Equal probability for opponents that have < ELOs than agent
                    for i,elo in enumerate(self.all_elos):
                        if elo < self.agent_elo:
                            weights[i] = 1
                    # Choose agent with highest ELO if agent ELO is higher than all opponent ELOs
                    if sum(weights) == 0:
                        weights[self.all_elos.index(min(self.all_elos))] = 1
            
        self.current_opp_idx = choice_with_normalization([i for i in range(len(self.all_elos))], weights)
        self.current_opp = self.all_opps[self.current_opp_idx]
        self.current_opp_elo = self.all_elos[self.current_opp_idx]
        #print("thread", self.my_port, "current opp elo:", self.current_opp_elo, "agent elo:", self.agent_elo, flush=True)
        self.env.load_new_opp(0, opp_fp(self.model_dir, self.current_opp), self.current_opp_elo)
        
    def get_agent_elo(self):
        return self.agent_elo

    def reset(self):
        if self.started:
            last_winner = self.env.last_winner
            if last_winner == 0:
                win_rate = 1.
                self.uncounted_games[0] += 1
            elif last_winner == 1:
                win_rate = 0.
                self.uncounted_games[1] += 1
            else:
                win_rate = .5
                
            agent_elo_change, _ = elo_change(self.agent_elo, self.current_opp_elo, self.K, win_rate)
            self.agent_elo += int(agent_elo_change)
            #print("THREAD", self.my_port, "CURRENT AGENT ELO:", self.agent_elo, flush=True)
        else:
            self.started = True
        
        self.next_opp()
        return self.env.reset()
        
    def step(self, action):
        if self.num_steps % self.elo_log_interval == 0:
            self.elo_log.append(self.agent_elo)
        self.num_steps += 1
        return self.env.step(action)
    
    def render(self, mode='console'):
        raise NotImplementedError()
        
    def close(self):
        self.env.close()