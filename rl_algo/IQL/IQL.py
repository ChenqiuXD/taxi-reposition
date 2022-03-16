import torch
import numpy as np
from utils.agent_q_function import AgentQFunction
from utils.base_agent import BaseAgent
import torch.nn as nn
import os
import copy


class IQL(BaseAgent):
    def __init__(self, args, env_config):
        """Agent using iql methods"""
        super().__init__(args, env_config)

        # Initialize buffer
        self.buffer_size = self.args.buffer_size
        self.buffer_ptr = 0
        if self.share_policy:
            self.buffer = np.zeros((self.buffer_size, 2 * self.dim_obs + 2))
        else:
            self.buffer = np.zeros((self.num_agents, self.buffer_size, 2 * self.dim_obs + 2))

        # Initialize agent networks
        agent_q_config = {
            'hidden_layers': [128, 128],  # hidden layers config for agent q function
            'dim_obs': self.dim_obs,
            'dim_action': self.dim_action
        }
        if self.share_policy:
            self.agent_q_nets = [AgentQFunction(self.args, agent_q_config).to(self.device)]
        else:
            self.agent_q_nets = [AgentQFunction(self.args, agent_q_config).to(self.device) for i in
                                 range(self.num_agents)]

        # Initialization
        for agent in self.agent_q_nets:
            for net in agent.modules():
                if isinstance(net, nn.Linear):
                    nn.init.xavier_uniform_(net.weight)
        # double dqn
        if self.args.use_double_q:
            print("Use double q learning")
            self.target_agent_q_nets = copy.deepcopy(self.agent_q_nets)
        # share policy
        print("Share policies: ", self.share_policy)

        # Collect all parameters
        self.parameters = []
        for agent in self.agent_q_nets:
            self.parameters += agent.parameters()

        self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        self.loss_func = nn.MSELoss()

        self.save_dir = os.path.abspath('.') + "\\IQL"

    def learn(self):
        self.prep_train()
        if self.share_policy:
            # sample batch from memory
            sample_index = np.random.choice(self.buffer_size, self.batch_size)
            batch_memory = self.buffer[sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.dim_obs]).to(self.device)
            batch_action = torch.LongTensor(batch_memory[:, self.dim_obs:self.dim_obs+1].astype(int)).to(self.device)
            batch_reward = torch.FloatTensor(batch_memory[:, self.dim_obs+1:self.dim_obs+2]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_obs:]).to(self.device)

            # q_eval
            q_eval = self.agent_q_nets[0](batch_state).gather(1, batch_action)
            q_next = self.target_agent_q_nets[0](batch_next_state).detach()
            q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            sample_index = np.random.choice(self.buffer_size, self.batch_size)
            batch_memory = [self.buffer[i][sample_index, :] for i in range(self.num_agents)]
            for i in range(self.num_agents):
                batch_state = torch.FloatTensor(batch_memory[i][:, :self.dim_obs]).to(self.device)
                batch_action = torch.LongTensor(batch_memory[i][:, self.dim_obs:self.dim_obs + 1].astype(int)).to(
                    self.device)
                batch_reward = torch.FloatTensor(batch_memory[i][:, self.dim_obs+1:self.dim_obs+2]).to(self.device)
                batch_next_state = torch.FloatTensor(batch_memory[i][:, -self.dim_obs:]).to(self.device)

                # calculate loss
                q_eval = self.agent_q_nets[i](batch_state).gather(1, batch_action)
                q_next = self.target_agent_q_nets[i](batch_next_state).detach()
                q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
                loss = self.loss_func(q_eval, q_target)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def hard_target_update(self):
        print("Hard update targets")
        for i in range(len(self.agent_q_nets)):
            self.target_agent_q_nets[i].load_state_dict(self.agent_q_nets[i].state_dict())

    def prep_train(self):
        for net in self.agent_q_nets:
            net.train()

    def prep_eval(self):
        for net in self.agent_q_nets:
            net.eval()

    def save_network(self, iter_cnt):
        for idx, net in enumerate(self.agent_q_nets):
            torch.save(net.state_dict(), self.save_dir+"\\net_param_"+str(idx) + "_iter_" + str(iter_cnt) + ".pkl")
            # torch.save(net.state_dict(), self.save_dir + "\\net_param_" + str(idx) + ".pkl")

    def restore(self):
        for idx, net in enumerate(self.agent_q_nets):
            path = self.save_dir+"\\net_param_"+str(idx)+".pth"
            net.load_state_dict(torch.load(path))
