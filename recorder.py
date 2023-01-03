import numpy as np
import datetime
import pickle

class Recorder:
    """ Recor necessary information. """
    def __init__(self, all_args) -> None:
        now = datetime.datetime.now()

        self.num_env_steps = all_args.num_env_steps
        self.episode_length = all_args.episode_length
        self.cnt = 0 

        self.file_name = all_args.algorithm_name \
                         + "_episodes_{}_length_{}_seed_{}_".format(self.num_env_steps, self.episode_length, all_args.seed)\
                         + now.strftime("%m_%d_%H_%M")

    def record_init_settings(self, setting):
        # setting: {"initial_drivers": self.env_config["initial_drivers"], 
        #  "upcoming_cars": self.env_config["upcoming_cars"], 
        #  "demands": self.env_config["node_demand"], 
        #  "edge_traffic": self.env_config["edge_traffic"]}
        self.init_setting = setting

    def record(self, reward_list, action_list, nodes_actions, idle_drivers):
        """ This function records the information during one episode """
        # Initialize the traj variable (require dimension of reward_list and action_list)
        if self.cnt==0:
            reward_types = reward_list[0].shape[0]
            self.reward_traj = np.zeros([self.num_env_steps, self.episode_length, reward_types])
            num_nodes = action_list[0].shape[0]
            self.bonus_traj = np.zeros([self.num_env_steps, self.episode_length, num_nodes])
            self.nodes_actions_traj = np.zeros([self.num_env_steps, self.episode_length, num_nodes, num_nodes])
            self.idle_drivers_traj = np.zeros([self.num_env_steps, self.episode_length, num_nodes])
        
        self.reward_traj[self.cnt] = np.array(reward_list)
        self.bonus_traj[self.cnt] = np.array(action_list)
        self.nodes_actions_traj[self.cnt] = np.array([nodes_actions])
        self.idle_drivers_traj[self.cnt] = idle_drivers
        self.cnt += 1
        if self.cnt > self.num_env_steps:
            raise RuntimeError("The recorder's cnt exceeds num_env_steps")

    def store_data(self):
        with open(self.file_name, 'wb') as f:
            data = {
                "init_setting": self.init_setting, 
                "reward_traj": self.reward_traj, 
                "bonus_traj": self.bonus_traj, 
                "nodes_actions_traj": self.nodes_actions_traj, 
                "idle_drivers_traj": self.idle_drivers_traj
            }
            pickle.dump(data, f)
        print("Saved file: " + self.file_name)