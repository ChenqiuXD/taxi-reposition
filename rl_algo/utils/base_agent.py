import numpy as np
import torch


class BaseAgent:
    def __init__(self, args, env_config):
        """Base agent class for q-methods ('iql', 'vdn', 'qmix') agents"""
        self.args = args
        self.algo = args.algorithm_name
        self.q_methods = ['iql', 'vdn', 'qmix']
        self.num_agents = env_config['num_agents']
        self.dim_action = env_config['dim_action']
        self.discrete_action_space = env_config['discrete_action_space']
        self.dim_obs = env_config['dim_observation']

        if self.algo not in self.q_methods:
            raise NotImplementedError("The method ", self.algo, 
                                      " is not in the q_methods. Please check base_agent.py to see details")

        # un-tunable parameters
        self.device = self.args.device
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.e_greedy = self.args.epsilon
        self.gamma = self.args.gamma
        self.batch_size = self.args.batch_size
        self.share_policy = args.share_policy
        print("Sharing policy: ", self.share_policy)

        # Buffer
        self.buffer_size = args.buffer_size
        self.buffer = []
        self.buffer_ptr = 0

    def learn(self):
        raise NotImplementedError

    def append_transition(self, obs, action, reward, obs_):
        """Store transition"""
        if self.share_policy:
            for i in range(self.num_agents):
                if self.buffer_ptr >= self.buffer_size:
                    self.buffer_ptr = 0
                self.buffer[self.buffer_ptr] = np.concatenate((obs[i], [action[i], reward[i]], obs_[i]))
                self.buffer_ptr += 1
        else:
            for i in range(self.num_agents):
                if self.buffer_ptr >= self.buffer_size:
                    self.buffer_ptr = 0
                self.buffer[i][self.buffer_ptr] = np.concatenate((obs[i], [action[i], reward[i]], obs_[i]))
            self.buffer_ptr += 1

    def choose_action(self, obs, is_random=False):
        """
        Choose action according to obs.
        :param obs: observation [1, 3*self.dim_obs]
        :param is_random: (bool) whether randomly choose action
        """
        eps = np.random.uniform(0, 1)
        if is_random or eps <= self.e_greedy:
            return [np.random.choice(range(self.dim_action)) for _ in range(self.num_agents)]
        else:
            if self.algo in self.q_methods:
                if self.share_policy:
                    obs = torch.FloatTensor(np.array(obs)).view(self.num_agents, self.dim_obs).to(self.device)
                    q_vals = self.agent_q_nets[0](obs)
                    actions = torch.max(q_vals, dim=1)[1].tolist()
                else:
                    actions = np.zeros(self.num_agents)
                    for i in range(self.num_agents):
                        q_vals = self.agent_q_nets[i](torch.FloatTensor(obs[i]).to(self.device))
                        _, action = torch.max(q_vals, dim=0)
                        actions[i] = action.item()
                return actions
            else:
                raise NotImplementedError("Method "+self.algo+" is not implemented")
