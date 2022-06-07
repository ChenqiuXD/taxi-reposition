# Implementation of DQN
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from rl_algo.utils.base_agent import BaseAgent

class QAgent(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.edge_index = env_config["edge_index"]
        self.len_mat = env_config["len_mat"]
        self.share_policy = True
        print("Using single agent q-learning, sharing policy is true.")
        self.q_net = QNetwork(n_features=self.dim_obs, n_agents=self.num_agents, episode_length=self.episode_length, l1=256, l2=128)
        for net in self.q_net.modules():
            if isinstance(net, nn.Linear):
                nn.init.normal_(net.weight, mean=0.0, std=0.05)
                nn.init.zeros_(net.bias)
        self.target_q_net = copy.deepcopy(self.q_net)

        #Buffer, a transition is [s, a, r, s', t], action is a [num_agent*1] vector, reward and time_step are 2 scalar
        self.buffer = np.zeros([self.buffer_size, self.dim_obs*2+self.num_agents+2])
        self.buffer_ptr = 0

        # Optimizer
        self.optim = torch.optim.SGD(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.save_dir = os.path.abspath('.')+"\\run_this\\agents\\q_learning\\"

    def append_transition(self, obs, action, reward, obs_):
        """Store transition"""
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate((self.to_state(obs), action, [reward], self.to_state(obs_), [obs['time_step']]))
        self.buffer_ptr += 1

    def learn(self):
        """Use samples from replay buffer to update q-network"""
        #sample batch from memory
        sample_index = np.random.choice(self.buffer_size, self.batch_size)
        batch_memory = self.buffer[sample_index, :]
        batch_state = batch_memory[:, :self.dim_obs]
        batch_action = torch.LongTensor(batch_memory[:, self.dim_obs:
                                        self.dim_obs+self.num_agents].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.dim_obs+self.num_agents:
                                                        self.dim_obs+self.num_agents+1])
        batch_next_state = batch_memory[:, -self.dim_obs-1:-1:]
        batch_time_step = batch_memory[:, -1].reshape(-1,1)
        batch_next_time_step = batch_time_step+1   # If batch_next_time_step==self.episode_length, then it means there is the last step

        # q_eval
        batch_state = np.concatenate((batch_state, batch_action, batch_time_step), axis=1)
        q_eval = self.q_net(torch.FloatTensor(batch_state))
        # q_target
        num_pow_actions = pow(self.dim_action, self.num_agents)
        concate_obs = np.concatenate((np.vstack([ [row]*num_pow_actions for row in batch_next_state ]),
                                      np.vstack([self.decode_action(i) for i in range(num_pow_actions)]*self.batch_size), 
                                      np.vstack([ [row]*num_pow_actions for row in batch_next_time_step])),
                                      axis=1)
        q_next = self.target_q_net(torch.FloatTensor(concate_obs)).detach().reshape(-1, num_pow_actions)
        q_next -= q_next*(batch_next_time_step==self.episode_length)    # For time_step==last_step, next_q=0 since it reaches final horizon. 
        q_target = batch_reward + q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_fn(q_eval, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def choose_action(self, obs, is_random=False):
        """Choose actions based on q-network"""
        actions = np.zeros(self.num_agents)
        if np.random.uniform()>=self.e_greedy or is_random:
            for i in range(self.num_agents):
                actions[i] = np.random.choice(np.arange(self.dim_action))
        else:
            num_pow_actions = pow(self.dim_action, self.num_agents)
            time_steps = np.array([obs['time_step']]*num_pow_actions).reshape(-1,1)
            obs = self.to_state(obs)
            concate_obs = np.concatenate((np.vstack([obs]*num_pow_actions),
                                          np.vstack([self.decode_action(i) for i in range(num_pow_actions)]), 
                                          time_steps),
                                          axis=1)
            action_value = self.q_net(torch.FloatTensor(concate_obs)).detach().numpy()
            action_list = np.where(action_value == np.max(action_value))[0]
            action_encoded = np.random.choice(action_list)
            actions = self.decode_action(action_encoded)
        return actions

    def decode_action(self, action):
        """transform action_array (scalar) to array"""
        assert action<=pow(self.dim_action, self.num_agents)
        action_array = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            action_array[i] = action % self.num_agents
            action = (action-action_array[i])/self.num_agents
        return action_array

    def to_state(self, obs):
        """This function transform state in dict form to numpy array (n_agents*dim_node_obs+n_edges*dim_edge_obs)"""
        if type(obs)==dict: # When obs is just one state
            state = np.concatenate([obs["idle_drivers"], 
                                    obs["upcoming_cars"], 
                                    obs["demands"], 
                                    np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])]), 
                                    self.len_mat])
        else:   # When obs is a batch of states
            state = np.vstack([np.concatenate([obs[i]["idle_drivers"], 
                                               obs[i]["upcoming_cars"], 
                                               obs[i]["demands"], 
                                               np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])]), 
                                               self.len_mat]) for i in range(len(obs))])
        return state

    def prep_train(self):
        self.q_net.train()
    
    def prep_eval(self):
        self.q_net.eval()

    def hard_target_update(self):
        print("Hard update targets")
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_network(self):
        import datetime
        file_name = "net_"+datetime.datetime.now().strftime('%m%d_%H%M')
        torch.save(self.q_net.state_dict(), self.save_dir+file_name+".pkl")
        print("Q-network saved in ", self.save_dir)

    def restore(self):
        self.q_net.load_state_dict(torch.load(self.save_dir+"net.pkl"))
        self.hard_target_update()
        print("Succesfully loaded q-network")

class QNetwork(nn.Module):
    def __init__(self, n_features, n_agents, episode_length, l1=256, l2=128):
        super(QNetwork, self).__init__()
        # Construct network - three layers
        # Network input should be : n_featuers + [5*1 agents' actions]
        self.fc1 = nn.Linear(n_features+n_agents+1, l1)    # input is [n_features + agents' action (1 for each agents)+time_step(1)]
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 1)     # output Q(s,a) a\in R^{n_agents}
        self.episode_length = episode_length

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x