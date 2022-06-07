from re import S
import numpy as np
import copy
import pickle
# import time

class Recorder:
    """Record the information during training"""
    def __init__(self, args, num_nodes) -> None:
        self.args = args
        self.episode_length = self.args.episode_length  
        self.num_episode = self.args.num_env_steps  # How many episode would be simulated
        self.num_nodes = num_nodes

        # Used to store the necessary transitions
        self.data_per_episodes = [0] * self.episode_length
        self.step_cnt = 0
        self.data_per_run = [0] * self.num_episode
        self.episode_cnt = 0
        
    def record_info_step(self, obs, reward_list, action, games, nodes_action, time_mat):
        """Record info during each step"""
        self.data_per_episodes[self.step_cnt] = {"obs": obs, "reward": reward_list, "action": action, "nodes actions": nodes_action, 
                                                 "nodes value": np.vstack([games[self.step_cnt].nodes[i].value_table for i in range(self.num_nodes)]),
                                                 "time mat": time_mat}
        self.step_cnt += 1
        if self.step_cnt>=self.episode_length+1:
            raise RuntimeError("Recorder record_info_step, step counter exceeds episode length")

    def record_info_episode(self, agent, games):
        """Record info at each episode"""
        self.data_per_run[self.episode_cnt] = copy.deepcopy(self.data_per_episodes)

        self.step_cnt = 0   # Refresh step counter
        self.episode_cnt += 1   
    
    def save_record(self):
        """Save pickle file to curret dir"""
        file_name = self.args.algorithm_name + "_episodes_{}_length_{}_seed_{}".format(self.num_episode, self.episode_length, self.args.seed)
        file = open(file_name, 'wb')
        pickle.dump(self.data_per_run, file)
        file.close()

        print("Done")
    
    def restore(self, result):
        """Restore the record so as to continue training
        @params: 
            result: (a list of dict) num_env_steps*episode_length, store the recorded observation
        """
        assert self.episode_length == len(result[0])    # Ensure that the loaded record has same episode length with current setting
        self.num_episode += len(result) # Plus the recorded result

        self.data_per_run = [0] * self.num_episode
        self.data_per_run[:len(result)] = result
        self.episode_cnt = len(result)