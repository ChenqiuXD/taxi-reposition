import numpy as np
from algorithms.base_agent import BaseAgent

class HeuristicPolicy(BaseAgent):
    def __init__(self, args, env_config):
        self.args = args
        self.env_config = env_config
        self.episode_length = args.episode_length
        self.max_bonus = env_config["dim_action"] - 1
        self.min_bonus = 0

        self.num_nodes = env_config["num_nodes"]
        self.nodes_policies = np.zeros([self.episode_length, self.num_nodes, self.num_nodes])

        self.cur_time_step = 0

    def append_transition(self, obs, action, reward, done, obs_, info):
        self.cur_time_step = obs["time_step"]
        nodes_actions = info[0]
        self.nodes_policies[self.cur_time_step] = nodes_actions

    def learn(self):
        for episode_idx in range(self.episode_length):
            for i in range(self.num_nodes):
                if self.ratio_list[episode_idx][i] > 4:
                    self.action[episode_idx][i] = 0
                elif self.ratio_list[episode_idx][i] > 3:
                    self.action[episode_idx][i] = 1
                elif self.ratio_list[episode_idx][i] >2:
                    self.action[episode_idx][i] = 2
                elif self.ratio_list[episode_idx][i] > 1:
                    self.action[episode_idx][i] = 3
                else:
                    self.action[episode_idx][i] = 4


    def choose_action(self, obs, is_random):
        """ directly return [0]*num_nodes """
        if is_random:
            return np.random.random([self.num_nodes])*( self.max_bonus -self.min_bonus ) + self.min_bonus
        else:
            time_step = obs["time_step"]
            return self.action[time_step]