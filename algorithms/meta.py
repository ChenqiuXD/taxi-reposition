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
        pass

    def learn(self):
        pass

    def choose_action(self, obs, is_random):
        """ directly return [0]*num_nodes """
        pass