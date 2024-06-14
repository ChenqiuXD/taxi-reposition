import numpy as np
from algorithms.base_agent import BaseAgent

class HeuristicPolicy(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        self.env_config = env_config
        self.episode_length = args.episode_length

        self.interval = args.lr
        self.min_bonus = args.min_bonus
        self.max_bonus = args.max_bonus

        self.num_nodes = env_config["num_nodes"]
        self.ratio_list = np.ones([self.episode_length, self.num_nodes])*1.01   # Multiply by 1.01, then bonuses would be initially zeros when applying the self.learn() function 
        self.action = np.ones([self.episode_length, self.num_nodes]) * self.min_bonus

    def append_transition(self, obs, action, reward, done, obs_, info):
        demands = obs["demands"]
        drivers = obs_["idle_drivers"]  # We use the bonuses at current time_step to influence the idle_drivers distribution at next time_step
        ratio = drivers / demands
        time_step = obs["time_step"]

        self.ratio_list[time_step] = ratio

    def learn(self):
        """ Change bonuses according to recorded idle_drivers/demands ratios. Note that the range of bonus are [-1, 1]"""
        for time_step in range(self.episode_length):
            for i in range(self.num_nodes):
                ratio = self.ratio_list[time_step][i]

                if ratio > 1.0:
                    self.action[time_step][i] = np.maximum(self.min_bonus, self.action[time_step][i]-self.interval)
                elif ratio < 1.0:
                    self.action[time_step][i] = np.minimum(self.max_bonus, self.action[time_step][i]+self.interval)

    def choose_action(self, obs, is_random=False):
        """ directly return [0]*num_nodes """
        if is_random:
            return np.random.random([self.num_nodes])*( self.max_bonus -self.min_bonus ) + self.min_bonus
        else:
            time_step = obs["time_step"]
            return self.action[time_step]