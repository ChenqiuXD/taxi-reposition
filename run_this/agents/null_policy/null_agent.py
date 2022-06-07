from rl_algo.utils.base_agent import BaseAgent
import numpy as np

class NullPolicy(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

    def choose_action(self, obs, is_random):
        return np.array([0]*self.num_agents)

    def append_transition(self, obs, action, reward, obs_):
        """No need to append transition. Used here to override"""
        pass

    def learn(self):
        """Random policy need not to learn. Used here to avoid NotImplementedError"""
        pass

    def save_network(self):
        pass