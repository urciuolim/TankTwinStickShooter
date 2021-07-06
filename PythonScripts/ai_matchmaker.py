import gym
from gym import spaces
from train import load_pop, load_stats
from gt_plot import sorted_keys, avg_elo
from preamble import gen_name
from human_matchmaking import opp_fp
from elo import elo_change
from tank_env import elo_based_choice, TankEnv

def new_ai_stats():
    return {
        "elo":[],
        "win_rate":{}
    }

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
                time_reward=-0.003
    ):
        super(AIMatchmaker, self).__init__()
        
        self.all_stats = all_stats
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
        
        self.started = False
        self.next_opp()
        
    def next_opp(self):
        self.current_opp_idx = elo_based_choice(self.all_elos, self.agent_elo, self.D)
        self.current_opp = self.all_opps[self.current_opp_idx]
        self.current_opp_elo = self.all_elos[self.current_opp_idx]
        #print("thread", self.my_port, "current opp elo:", self.current_opp_elo, flush=True)
        self.env.load_new_opp(0, opp_fp(self.model_dir, self.current_opp), self.current_opp_elo)
        
    def get_agent_elo(self):
        return self.agent_elo

    def reset(self):
        if self.started:
            last_winner = self.env.last_winner
            if last_winner == 0:
                win_rate = 1.
            elif last_winner == 1:
                win_rate = 0.
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
        return self.env.step(action)
    
    def render(self, mode='console'):
        raise NotImplementedError()
        
    def close(self):
        self.env.close()