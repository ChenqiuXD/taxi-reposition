from rl_algo.utils.base_agent import BaseAgent
# from utils.base_agent import BaseAgent

class RandPolicy(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

    def choose_action(self, obs, is_random):
        return super().choose_action(obs, is_random=True)

    def append_transition(self, obs, action, reward, obs_):
        """No need to append transition. Used here to override"""
        pass

    def learn(self):
        """Random policy need not to learn. Used here to avoid NotImplementedError"""
        pass