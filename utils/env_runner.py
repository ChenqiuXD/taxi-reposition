import numpy as np
from utils.plot import plot_result
import os
from tqdm import tqdm


class EnvRunner:
    """Environment runner for overall simulation."""
    def __init__(self, args, env, recorder):
        # non-tunable hyper-parameters
        self.args = args
        self.device = args.device
        self.recorder = recorder

        # set tunable hyper-parameters
        self.algorithm_name = self.args.algorithm_name
        self.num_env_steps = self.args.num_env_steps

        # eval, save, log interval
        self.use_eval = self.args.use_eval
        self.train_interval_episode = self.args.train_interval_episode
        # self.eval_interval = self.args.eval_interval
        # self.save_interval = self.args.save_interval
        # self.log_interval = self.args.log_interval

        # eval, save, log time (T)
        self.total_train_steps = 0  # number of gradient updates performed
        self.total_env_steps = 0    # number of total experienced transition
        self.last_train_T = 0   # last train step
        # self.last_eval_T = 0  # last episode after which a eval run was conducted
        # self.last_save_T = 0  # last epsiode after which the models were saved
        # self.last_log_T = 0
        self.episode_length = self.args.episode_length
        # self.eval_interval = self.episode_length * 100

        self.env = env
        self.num_nodes = env.num_nodes

        # setup the agent 
        self.agent = self.setup_agent()

    def setup_agent(self):
        """Setup agent algorithm"""
        # initialize the policies and organize the agents
        if self.algorithm_name == "null":
            from algorithms.null import NullPolicy as Algo
        elif self.algorithm_name == 'heuristic':
            from algorithms.heuristic import HeuristicPolicy as Algo
        elif self.algorithm_name == 'direct':
            from algorithms.direct import DirectPolicy as Algo
        elif self.algorithm_name == 'ddpg':
            from algorithms.ddpg import DDPG as Algo
            # from algorithms.ddpg_calc import DDPG_calc as Algo
        elif self.algorithm_name == 'ppo':
            from algorithms.ppo import PPO as Algo
        elif self.algorithm_name == 'ddpg_graph':
            from algorithms.ddpg_graph import DDPG_graph as Algo
        elif self.algorithm_name == 'metaGrad':
            from algorithms.metaGrad import metaAgent as Algo
        else:
            raise NotImplementedError("The method " + self.algorithm_name + " is not implemented. Please check env_runner.py line 40 to see whether your method is added to the setup_agent function ")

        env_config = {
            "num_nodes": self.env.num_nodes,
            "dim_action": self.env.num_nodes,       # bonuses, thus dimension of action should be num of nodes. 
            "dim_observation": self.env.obs_spaces,  # Full observation (equals num_nodes*dim_node_obs+num_edges*dim_edge_obs)
            "dim_node_obs": self.env.dim_node_obs,  # node's observation 3, [idle drivers, upcoming cars, demands]
            "dim_edge_obs": self.env.dim_edge_obs,  # edge's observation 2, [edge_traffic, len_mat]
            "edge_index": self.env.edge_index, 
            "len_mat": self.env.len_mat, 
        }
        return Algo(self.args, env_config)

    def warmup(self):
        """ Warmup drivers to equilibrium given minimum bonuses """
        print("Warming up the drivers")
        for _ in tqdm( range( 1, self.args.warmup_steps ) ):
            obs = self.env.reset()
            while True:
                action = np.ones(self.num_nodes) * (self.args.min_bonus + self.args.max_bonus) / 2
                obs_, reward_list, done, info = self.env.step(action)  # Assign false to make drivers update policies. 
                obs = obs_
                if done:
                    obs = self.env.reset()
                    break

    def run(self):
        """Collect a training episode transitions and perform training, saving, logging, and evaluation steps"""
        self.agent.prep_train()     # call all network.train()

        # initial observation
        obs = self.env.reset()
        reward_traj = []
        action_traj = np.zeros([self.episode_length, self.num_nodes])
        nodes_actions_traj = np.zeros([self.episode_length, self.num_nodes, self.num_nodes])
        idle_drivers_traj = np.zeros([self.episode_length, self.num_nodes])
        step = 0
        while step < self.episode_length:
            # if self.args.render:
            #     self.env.render(mode="not_human")

            action = self.agent.choose_action(obs)                
            obs_, reward_list, done, info = self.env.step(action)  # reward_list : [idle_prob, avg_travelling_cost, bonuses_cost, overall_cost](+,+,+,-) only the last one is minus
            # print("At step", step, " agent choose action ", np.round(action, decimals=3))
            # self.env.decrease_lr(step)  # Decrease the learning rate of drivers' agents
            # print("At step {}, costs are: idle_prob {}, travelling_cost {}, bonuses_cost {}".format(step, reward_list[0], reward_list[1], reward_list[2]) )
            
            # agent update
            self.agent.append_transition(obs, action, reward_list[-1], done, obs_, info)
            if self.last_train_T == 0 or ((self.total_env_steps-self.last_train_T) / self.train_interval_episode >= 1):
                self.agent.learn()
                self.total_train_steps += 1
                self.last_train_T = self.total_env_steps

            # Append to record list 
            reward_traj.append(reward_list)
            action_traj[step] = action
            nodes_actions_traj[step] = self.env.get_nodes_actions(step)
            idle_drivers_traj[step] = obs_["idle_drivers"]

            obs = obs_
            self.total_env_steps += 1
            step += 1

            if np.all(done):
                break

        self.recorder.record(reward_traj, action_traj, nodes_actions_traj, idle_drivers_traj)
        return reward_traj

    def store_data(self):
        """Store necessary data"""
        os.makedirs(self.args.output_path, exist_ok=True)
        self.agent.save_model(self.args.output_path)

        data = self.recorder.store_data()
        plot_result(self.args, data)