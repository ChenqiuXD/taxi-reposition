import numpy as np
from algorithms.base_agent import BaseAgent

class HeuristicPolicy(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        self.env_config = env_config
        self.episode_length = args.episode_length

        self.n_interval = 10
        self.min_bonus = args.min_bonus
        self.max_bonus = args.max_bonus
        self.bonus_degree = np.linspace(self.max_bonus, self.min_bonus, self.n_interval)
        self.min_ratio = 0.5
        self.max_ratio = 1.0
        self.ratio_degree = np.linspace(self.min_ratio, self.max_ratio, self.n_interval)

        self.num_nodes = env_config["num_nodes"]
        self.ratio_list = np.ones([self.episode_length, self.num_nodes])*1.01   # Multiply by 1.01, then bonuses would be initially zeros when applying the self.learn() function 
        self.action = np.zeros([self.episode_length, self.num_nodes])
        self.cur_time = 0

    def append_transition(self, obs, action, reward, done, obs_, info):
        demands = obs["demands"]
        drivers = obs_["idle_drivers"]  # We use the bonuses at current time_step to influence the idle_drivers distribution at next time_step
        ratio = drivers / demands
        time_step = obs["time_step"]

        self.ratio_list[time_step] = ratio

    def learn(self):
        """ Change bonuses according to recorded idle_drivers/demands ratios. Note that the range of bonus are [-1, 1]"""
        for time_step in range(self.episode_length):
            for i in range(self.episode_length):
                ratio = self.ratio_list[time_step][i]

                if ratio > self.max_ratio:
                    self.action[time_step][i] = self.min_bonus
                elif ratio < self.min_ratio:
                    self.action[time_step][i] = self.max_bonus
                else:
                    loc = np.where(ratio<self.ratio_degree)[0][0] # Find which interval that current ratio locates
                    self.action[time_step][i] = self.bonus_degree[loc]

    def choose_action(self, obs, is_random=False):
        """ directly return [0]*num_nodes """
        if is_random:
            return np.random.random([self.num_nodes])*( self.max_bonus -self.min_bonus ) + self.min_bonus
        else:
            time_step = obs["time_step"]
            return self.action[time_step]