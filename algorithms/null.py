import numpy as np
from algorithms.base_agent import BaseAgent

class NullPolicy(BaseAgent):
    def __init__(self, args, env_config):
        self.args = args
        self.env_config = env_config

        self.num_nodes = env_config["num_nodes"]

    def choose_action(self, obs, is_random):
        """ directly return [-1]*num_nodes. Note that the action space range from [-1, 1] """
        return np.array([-1]*self.num_nodes)