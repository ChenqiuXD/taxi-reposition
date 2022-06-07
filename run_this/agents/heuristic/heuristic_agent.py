from rl_algo.utils.base_agent import BaseAgent
import numpy as np

class HeuristicAgent(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.episode_length = args.episode_length
        self.num_agents = env_config["num_agents"]
        self.dim_action = env_config["dim_action"]  # 5, since bonus range from [0,1,2,3,4]
        self.edge_index = env_config["edge_index"]
        self.bonus_table = np.zeros([self.episode_length, self.num_agents]) # Action list of heuristic agent. 
        self.obs_list = [0]*(self.episode_length+1)
        self.cur_time = 0

    def choose_action(self, obs, is_random):
        if is_random:
            return [np.random.choice(range(self.dim_action)) for _ in range(self.num_agents)]
        else:
            return self.bonus_table[obs["time_step"]]

    def append_transition(self, obs, action, reward, obs_):
        """No need to append transition. Used here to override"""
        self.cur_time = obs["time_step"]
        self.obs_list[self.cur_time] = obs
        self.obs_list[self.cur_time+1] = obs_

    def learn(self):
        """
        Heursitically assign bonus according to experience. 
        !!! Pay attention that heuristic agent must learn at each step!!!
        """
        if self.cur_time == self.episode_length-1:  # Learn require idle_drivers at latter state, thus would skip when cur_tim=0
            pass
        
        demand = self.obs_list[self.cur_time]["demands"]
        idle_drivers = self.obs_list[self.cur_time+1]["idle_drivers"]
        for i in range(self.num_agents):
            if idle_drivers[i]==0:
                self.bonus_table[self.cur_time, i] = self.dim_action-1                  # 5-1 = 4
            elif idle_drivers[i] <= demand[i]/4:
                self.bonus_table[self.cur_time, i] = np.maximum(0, self.dim_action-2)   # 5-2 = 3
            elif idle_drivers[i] <= demand[i]/4*2:
                self.bonus_table[self.cur_time, i] = np.maximum(0, self.dim_action-3)   # 5-3 = 2
            elif idle_drivers[i] <= demand[i]/4*3:
                self.bonus_table[self.cur_time, i] = np.maximum(0, self.dim_action-4)   # 5-4 = 1
            else:
                self.bonus_table[self.cur_time, i] = np.maximum(0, self.dim_action-5)   # 5-5 = 0
    
    def save_network(self):
        """Heuristic do not need save. """
        pass
        