import numpy as np
# from recorder import Recorder
import pickle

class EnvRunner:
    """Environment runner for overall simulation."""
    def __init__(self, args, env):
        # non-tunable hyper-parameters
        self.args = args
        self.device = args.device

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
        """Setup agent algorithm
        P.S. setup this function so that in final experiment, new algorithms could be added"""
        # initialize the policies and organize the agents
        if self.algorithm_name == "null":
            from algorithms.null import NullPolicy as Algo
        elif self.algorithm_name == 'heuristic':
            from algorithms.heuristic import HeuristicPolicy as Algo
        else:
            raise NotImplementedError("The method " + self.algorithm_name + " is not implemented. Please check env_runner.py line 40 to see whether your method is added to the setup_agent function ")

        env_config = {
            "num_nodes": self.env.num_nodes,
            "dim_action": self.env.num_nodes,       # 5, since there are 5 nodes
            "dim_observation": self.env.obs_spaces,  # Full observation (equals num_nodes*dim_node_obs+num_edges*dim_edge_obs)
            "dim_node_obs": self.env.dim_node_obs,  # node's observation 3, [idle drivers, upcoming cars, demands]
            "dim_edge_obs": self.env.dim_edge_obs,  # edge's observation 2, [edge_traffic, len_mat]
            "edge_index": self.env.edge_index, 
            "len_mat": self.env.len_mat, 
        }
        return Algo(self.args, env_config)

    def warmup(self):
        """Fill up agents' replay buffer"""
        # Warmup the environment first
        print("Warming up the drivers")
        for _ in range(self.args.max_warmup_steps):
            obs = self.env.reset()
            while True:
                action = np.zeros(self.num_agent)
                obs_, reward_list, done, info = self.env.step(action, is_warmup=False)  # Assign false to make drivers update policies. 
                obs = obs_
                if done:
                    obs = self.env.reset()
                    break

        if self.algorithm_name not in self.warmup_methods:   # Other algorithms do not require warmup
            print("algorithm {} do not require warmup. ".format(self.algorithm_name))
            return

        print("Currently warming up")
        episode_num = int(self.args.buffer_size/self.episode_length)+1
        for i in range(episode_num):
            obs = self.env.reset()
            num_steps = self.episode_length if episode_num!=1 else self.args.buffer_size
            for j in range(num_steps):
                # action = self.agent.choose_action(obs, is_random=True)
                action = np.zeros(self.num_agent)
                obs_, reward_list, done, info = self.env.step(action, is_warmup=True)
                self.agent.append_transition(obs, action, reward_list[-1], done, obs_, info)
                obs = obs_
                if np.all(done):
                    obs = self.env.reset()
                print("Episode {}/{}, iteration {}/{}".format(i+1, int(self.args.buffer_size/self.episode_length)+1, j+1, num_steps))
        print("Finished warming up")

    def run(self):
        """Collect a training episode and perform training, saving, logging and evaluation steps"""
        # Collect data
        self.agent.prep_train()     # call all network.train()

        # initial observation
        obs = self.env.reset()
        reward_traj = []
        action_traj = []
        step = 0
        while step < self.episode_length:
            if self.args.render:
                self.env.render(mode="not_human")

            action = self.agent.choose_action(obs, is_random=False)
            action = self.normalize_actions(action)
            obs_, reward_list, done, info = self.env.step(action)  # reward_list : [idle_prob, avg_travelling_cost, bonuses_cost, overall_cost](+,+,+,-) only the last one is minus
            print("At step", step, " agent choose action ", action)
            print("At step {}, costs are: idle_prob {}, travelling_cost {}, bonuses_cost {}".format(step, reward_list[0], reward_list[1], reward_list[2]) )
            reward_traj.append(reward_list)
            action_traj.append(action)
            
            self.agent.append_transition(obs, action, reward_list[-1], done, obs_, info)
            if self.last_train_T == 0 or ((self.total_env_steps-self.last_train_T) / self.train_interval_episode >= 1):
                self.agent.learn()
                self.total_train_steps += 1
                self.last_train_T = self.total_env_steps

            # if (self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval >= 1:
            #     self.last_hard_update_T = self.total_env_steps
            #     self.agent.hard_target_update()

            obs = obs_
            self.total_env_steps += 1
            step += 1

            if np.all(done):
                break

        return reward_traj, action_traj

    def normalize_actions(self, actions):
        """ Normalize the actions from [-1, 1] to [min_bonus, max_bonus] """
        k = (self.args.max_bonus - self.args.min_bonus) / (1-(-1))
        return (actions-(-1))*k+self.args.min_bonus

    def store_data(self):
        """This function store necessary data"""
        pass

    def restore(self, isEval=False):
        pass
