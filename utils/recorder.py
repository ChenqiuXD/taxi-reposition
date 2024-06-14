import numpy as np
# import datetime
import pickle

import os

class Recorder:
    """ Record necessary information. """
    def __init__(self, all_args) -> None:
        """ Initialize the recorder with the arguments."""
        self.output_path = all_args.output_path
        self.mode = all_args.mode

        self.algorithm_name = all_args.algorithm_name
        self.num_env_steps = all_args.num_env_steps
        self.episode_length = all_args.episode_length
        self.cnt = 0 

        self.file_name = all_args.algorithm_name \
                         + "_episodes_{}_length_{}_seed_{}_".format(self.num_env_steps, self.episode_length, all_args.seed)

    def record_init_settings(self, setting):
        self.init_setting = setting

    def record(self, reward_list, action_list, nodes_actions, idle_drivers):
        """ Records the information of the current episode into the array."""
        # Initialize the traj variable (require dimension of reward_list and action_list)
        reward_types = reward_list[0].shape[0]
        num_nodes = action_list[0].shape[0]
        if self.cnt==0 and not hasattr(self, "reward_traj"):    # Means mode is 'train'
            self.reward_traj = np.zeros([self.num_env_steps, self.episode_length, reward_types])
            self.bonus_traj = np.zeros([self.num_env_steps, self.episode_length, num_nodes])
            self.nodes_actions_traj =np.zeros([self.num_env_steps, self.episode_length, num_nodes, num_nodes])
            self.idle_drivers_traj = np.zeros([self.num_env_steps, self.episode_length, num_nodes])
        
        self.reward_traj[self.cnt] = np.array(reward_list)
        self.bonus_traj[self.cnt] = np. array(action_list)
        self.nodes_actions_traj[self.cnt] = np.array([nodes_actions])
        self.idle_drivers_traj[self.cnt] = idle_drivers
        self.cnt += 1

        if self.cnt > self.num_env_steps:
            raise RuntimeError("The recorder's cnt exceeds num_env_steps")

    def store_data(self):
        """ Store the data into a pickle file."""
        data = {
            "init_setting": self.init_setting, 
            "reward_traj": self.reward_traj, 
            "bonus_traj": self.bonus_traj, 
            "nodes_actions_traj": self.nodes_actions_traj, 
            "idle_drivers_traj": self.idle_drivers_traj
        }
        with open(os.path.join(self.output_path, self.file_name), 'wb') as f:
            pickle.dump(data, f)
        print("Saved file: " + self.output_path)

        return data