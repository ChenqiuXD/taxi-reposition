import numpy as np
import torch


class Runner:
    def __init__(self, args, env):
        """
        Used to train algorithms.
        :param config: (dict) config dictionary containing parameters for training
        """
        # non-tunable hyper-parameters
        self.args = args
        self.device = args.device

        # set tunable hyper-parameters
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        # self.per_alpha = self.args.per_alpha
        # self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval = self.args.hard_update_interval

        # eval, save, log interval
        self.use_eval = self.args.use_eval
        self.train_interval_episode = self.args.train_interval_episode
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        # eval, save, log time (T)
        self.total_train_steps = 0  # number of gradient updates performed
        self.total_env_steps = 0    # number of total experienced transition
        self.last_train_T = 0   # last train step
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_T = 0
        self.episode_length = self.args.episode_length
        self.eval_interval = self.episode_length * 100

        self.env = env
        self.num_agent = env.n

        # initialize the policies and organize the agents
        if self.algorithm_name == "qmix":
            from QMix.q_mixer import QMixer as Algo
        elif self.algorithm_name == 'iql':
            from IQL.IQL import IQL as Algo
        elif self.algorithm_name == 'vdn':
            from VDN.vdn import VDN as Algo
        else:
            raise NotImplementedError("The method "+self.algorithm_name+" is not implemented")

        env_config = {
            "num_agents": self.env.n,
            "dim_action": self.env.action_space[0].n,
            "dim_observation": self.env.observation_space[0].shape[0],
            "discrete_action_space": self.env.discrete_action_space
        }
        self.agent = Algo(self.args, env_config)
        if self.args.train:
            self.warmup()

    def warmup(self):
        """Fill up agents' replay buffer"""
        for i in range(int(self.args.buffer_size/self.episode_length)+1):
            obs = self.env.reset()
            for _ in range(self.episode_length):
                action = self.agent.choose_action(obs, is_random=True)
                obs_, reward, done, _ = self.env.step(action)
                self.agent.append_transition(obs, action, reward, obs_)
                if np.all(done):
                    obs = self.env.reset()
        print("Finished warming up")

    def run(self):
        """Collect a training episode and perform training, saving, logging and evaluation steps"""
        # Collect data
        self.agent.prep_train()     # call all network.train()

        # initial observation
        obs = self.env.reset()
        reward_list = []
        step = 0
        while step < self.episode_length:
            if self.args.render:
                self.env.render(mode="not_human")
            action = self.agent.choose_action(obs, is_random=False)
            # print("At step ", step, " agent choose action ", action)
            obs_, reward, done, _ = self.env.step(action)
            reward_list.append(reward[0])

            self.agent.append_transition(obs, action, reward, obs_)
            if self.last_train_T == 0 or ((self.total_env_steps-self.last_train_T) / self.train_interval_episode >= 1):
                self.agent.learn()
                self.total_train_steps += 1
                self.last_train_T = self.total_env_steps

            if (self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval >= 1:
                self.last_hard_update_T = self.total_env_steps
                self.agent.hard_target_update()

            obs = obs_
            self.total_env_steps += 1
            step += 1

            if np.all(done):
                break

        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # save
        if ((self.total_env_steps - self.last_save_T) / self.save_interval) >= 1:
            self.save(self.total_env_steps)
            self.last_save_T = self.total_env_steps

        # eval
        # if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
        #     self.agent.prep_eval()
        #     self.eval()
        #     self.last_eval_T = self.total_env_steps

        return reward_list

    def save(self, iter_cnt):
        """Save the network"""
        print("Saving network at total_num_step: ", self.total_env_steps)
        self.agent.save_network(iter_cnt)

    def log(self):
        """Log necessary information"""
        pass

    def eval(self):
        """Evaluate the policy"""
        self.agent.prep_eval()

        obs = self.env.reset()
        step = 0
        reward_list = []
        while step <= self.episode_length:
            self.env.render(mode='not_human')
            action = self.agent.choose_action(obs, is_random=False)
            obs_, reward, done, _ = self.env.step(action)
            reward_list.append(reward[0])
            obs = obs_
            step += 1

            if np.all(done):
                break
        return reward_list

    def restore(self):
        """Load network parameters"""
        self.agent.restore()
