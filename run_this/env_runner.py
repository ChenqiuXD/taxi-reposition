from rl_algo.utils.runner import Runner
import numpy as np
from recorder import Recorder
import pickle

class EnvRunner(Runner):
    """Environment runner for overall simulation."""
    def __init__(self, args, env):
        self.q_methods = ['q_learning', 'iql', 'vdn', 'mvdn', 'mq_learning']   # Assigned fisrt since is used in super().__init__()
        self.model_based_methods = ['mvdn', 'mq_learning']
        super().__init__(args, env)
        self.recorder = Recorder(args, num_nodes=self.num_agent)

    def setup_agent(self):
        """Setup agent algorithm
        P.S. setup this function so that in final experiment, new algorithms could be added"""
        # initialize the policies and organize the agents
        if self.algorithm_name == "qmix":
            from rl_algo.QMix.q_mixer import QMixer as Algo
        elif self.algorithm_name == 'iql':
            # from agents.iql.iql import IQLAgent as Algo
            pass
        elif self.algorithm_name == 'vdn':
            from agents.vdn.vdn import VDN as Algo
        elif self.algorithm_name == 'random':
            from agents.random_policy.rand_policy import RandPolicy as Algo
        elif self.algorithm_name == 'null':
            from agents.null_policy.null_agent import NullPolicy as Algo
        elif self.algorithm_name == "q_learning":
            from agents.q_learning.dqn import QAgent as Algo
        elif self.algorithm_name == "heuristic":
            from agents.heuristic.heuristic_agent import HeuristicAgent as Algo
        elif self.algorithm_name == "mvdn":
            from agents.model_vdn.model_vdn import M_VDN as Algo
        else:
            raise NotImplementedError("The method "+self.algorithm_name+" is not implemented")

        env_config = {
            "num_agents": self.env.n_node,
            "dim_action": self.env.action_space,    # 5, since bonus range from [0,1,2,3,4]
            "dim_observation": self.env.obs_spaces,  # Full observation (equals num_agents*dim_node_obs+num_edges*dim_edge_obs)
            "dim_node_obs": self.env.dim_node_obs,  # node's observation 3, [idle drivers, upcoming cars, demands]
            "dim_edge_obs": self.env.dim_edge_obs,  # edge's observation 2, [edge_traffic, len_mat]
            "discrete_action_space": self.env.action_space, # Same as "dim_action". (Used mainly in gym, thus herited here. But actually is useless)
            "edge_index": self.env.edge_index, 
            "len_mat": self.env.len_mat, 
        }
        return Algo(self.args, env_config)

    def warmup(self):
        """Fill up agents' replay buffer"""
        if self.algorithm_name not in self.q_methods:   # Other algorithms do not require warmup
            return

        print("Currently warming up")
        episode_num = int(self.args.buffer_size/self.episode_length)+1
        for i in range(episode_num):
            obs = self.env.reset()
            num_steps = self.episode_length if episode_num!=1 else self.args.buffer_size
            for j in range(num_steps):
                action = self.agent.choose_action(obs, is_random=True)
                obs_, reward_list, done, _ = self.env.step(action, is_warmup=True)
                if self.algorithm_name in self.model_based_methods:
                    self.agent.append_transition(obs, action, reward_list[-1], obs_, self.env.sim_time_mat) 
                else: 
                    self.agent.append_transition(obs, action, reward_list[-1], obs_)
                obs = obs_
                if np.all(done):
                    obs = self.env.reset()
                print("Episode {}/{}, iteration {}/{}".format(i+1, int(self.args.buffer_size/self.episode_length)+1, j+1, num_steps))

                if self.args.episode_length==1:
                    for _ in range(self.args.buffer_size):
                        action = self.agent.choose_action(obs, is_random=True)
                        if self.algorithm_name in self.model_based_methods:
                            self.agent.append_transition(obs, action, reward_list[-1], obs_, self.env.sim_time_mat) 
                        else: 
                            self.agent.append_transition(obs, action, reward_list[-1], obs_)
                    print("Episode_length==1, thus repeated added transition")
                    return
        print("Finished warming up")

    def run(self):
        """Collect a training episode and perform training, saving, logging and evaluation steps"""
        # Collect data
        self.agent.prep_train()     # call all network.train()

        # initial observation
        obs = self.env.reset()
        reward_traj = []
        step = 0
        while step < self.episode_length:
            if self.args.render:
                self.env.render(mode="not_human")
            print("Current idle_drivers'distribution is: \n", obs["idle_drivers"])
            action = self.agent.choose_action(obs, is_random=False)
            obs_, reward_list, done, _ = self.env.step(action)  # reward_list : [idle_prob, avg_travelling_cost, bonuses_cost, overall_cost](+,+,+,-) only the last one is minus
            print("At step", step, " agent choose action ", action)
            print("At step {}, costs are: idle_prob {}, travelling_cost {}, bonuses_cost {}".format(step, reward_list[0], reward_list[1], reward_list[2]) )
            reward_traj.append(reward_list)
            
            if self.algorithm_name in self.model_based_methods: # model-based methods need to store time_mat
                self.agent.append_transition(obs, action, reward_list[-1], obs_, self.env.sim_time_mat)
            else:
                self.agent.append_transition(obs, action, reward_list[-1], obs_)
            if self.last_train_T == 0 or ((self.total_env_steps-self.last_train_T) / self.train_interval_episode >= 1):
                self.agent.learn()
                self.total_train_steps += 1
                self.last_train_T = self.total_env_steps

            if (self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval >= 1:
                self.last_hard_update_T = self.total_env_steps
                self.agent.hard_target_update()

            # Record trajectory during steps
            self.recorder.record_info_step(obs, reward_list, action, self.env.games, self.env.nodes_action_result, self.env.sim_time_mat)

            obs = obs_
            self.total_env_steps += 1
            step += 1

            if np.all(done):
                break
        self.recorder.record_info_episode(self.agent, self.env.games)

        return reward_traj

    def store_data(self):
        """This function store necessary data"""
        self.recorder.save_record()
        self.agent.save_network()

    def restore(self, isEval=False):
        """This function restore agent, env, and recorder
        @params:
            isEnv: (Bool) In evaluation, we do not restore environemnt and recorder. 
        """
        # Restore agent: load network parameters
        try:
            if self.args.algorithm_name in self.q_methods:
                self.agent.restore()
            else:
                print("Method ", self.args.algorithm_name, " does not need restore")
        except:
            raise RuntimeError("Agent do not have restore function or restore has failed ")
        
        # Read recorded result to restore recorder and env
        if not isEval:
            import os
            last_step_cnt = 300
            try:
                path = os.path.abspath('.')+"\\"+self.args.algorithm_name+"_episodes_"+str(last_step_cnt)+"_length_"+str(self.args.episode_length)
                f = open(path, 'rb')
                result = pickle.load(f)
            except:
                print("Unable to find result file, restore aborted")

            # Restore env
            self.env.restore(result)    # result[-1] is a list of dict(observation)

            # Restore recorder
            self.recorder.restore(result)
