import copy
import torch.nn as nn
import torch.optim
from utils.agent_q_function import AgentQFunction
from utils.base_agent import BaseAgent
import numpy as np
import os


class VDN(BaseAgent):
    """
    Trainer class for QMix with MLP policies.
    """
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

        # Initialize buffer
        self.buffer_size = self.args.buffer_size
        self.buffer_ptr = 0
        self.buffer = np.zeros((self.buffer_size, 2*self.dim_obs*self.num_agents+2*self.num_agents))

        # Initialize mixer network
        self.share_policy = self.args.share_policy

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

        for agent in self.agent_q_nets:
            for net in agent.modules():
                if isinstance(net, nn.Linear):
                    nn.init.xavier_uniform_(net.weight)
        # double dqn
        if self.args.use_double_q:
            print("Using double q learning")
            self.target_agent_q_nets = copy.deepcopy(self.agent_q_nets)
        else:
            raise NotImplementedError("Performance of not using double dqn is so poor thus is not implemented")

        # Collect all parameters
        self.parameters = []
        for agent in self.agent_q_nets:
            self.parameters += agent.parameters()

        self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        self.loss_func = nn.MSELoss()

        self.save_dir = os.path.abspath('.') + "\\VDN"

    def learn(self):
        """Update the network parameters using Q-learning"""
        self.prep_train()
        if self.share_policy:
            sample_index = np.random.choice(self.buffer_size, self.batch_size)
            batch_memory = self.buffer[sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.dim_obs*self.num_agents]).to(self.device)
            batch_action = torch.LongTensor(
                batch_memory[:, self.dim_obs*self.num_agents: (self.dim_obs+1)*self.num_agents]).to(self.device)
            batch_reward = torch.FloatTensor(
                batch_memory[:, (self.dim_obs+1)*self.num_agents: (self.dim_obs+1)*self.num_agents+1]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_obs*self.num_agents:]).to(self.device)

            # Current Q values
            curr_agent_q_vals = self.get_q_value(batch_state, action=batch_action, is_target=False)
            curr_mixed_q = torch.sum(curr_agent_q_vals, dim=1).view(-1, 1)

            # Next state Q values
            next_agent_q_vals_all = self.get_q_value(batch_next_state, is_target=True)
            next_agent_q_vals, _ = torch.max(next_agent_q_vals_all, dim=2)
            next_mixed_q = torch.sum(next_agent_q_vals, dim=1).view(-1, 1)

            # Compute Bellman loss
            target_mixed_q = (batch_reward + self.gamma * next_mixed_q).detach()
            loss = self.loss_func(curr_mixed_q, target_mixed_q)
            # print("Current loss is: ", loss)

            # optimise
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            sample_index = np.random.choice(self.buffer_size, self.batch_size)
            batch_memory = self.buffer[sample_index, :]

            # Get agent q-vals
            batch_state = torch.FloatTensor(batch_memory[:, :self.dim_obs*self.num_agents]).to(self.device)
            batch_action = torch.LongTensor(
                batch_memory[:, self.dim_obs*self.num_agents: (self.dim_obs+1)*self.num_agents]).to(self.device)
            batch_reward = torch.FloatTensor(
                batch_memory[:, (self.dim_obs+1)*self.num_agents: (self.dim_obs+1)*self.num_agents+1]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_obs*self.num_agents:]).to(self.device)

            # Current Q values
            curr_q_vals = self.get_q_value(batch_state, action=batch_action, is_target=False)
            curr_mixed_q = torch.sum(curr_q_vals, dim=1)

            # Next Q values
            next_q_vals = self.get_q_value(batch_next_state, is_target=True)
            next_agent_q_vals, _ = torch.max(next_q_vals, dim=2)
            next_mixed_q = torch.sum(next_agent_q_vals, dim=1).view(-1, 1)

            # Compute Bellman loss
            target_mixed_q = (batch_reward + self.gamma * next_mixed_q).detach().view(-1)
            loss = self.loss_func(curr_mixed_q, target_mixed_q)

            # optimise
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_q_value(self, batch_obs, action=None, is_target=False):
        """Get the batch q value of state-action pair
        :param: batch_obs is the observation (dim: [batch_size, num_agents*dim_obs])
        :param: action is the batch of actions (dim: [batch_size, num_agents])
        :param: is_target defines whether get q values from target network
        """
        if isinstance(batch_obs, np.ndarray):   # Only used when choosing action
            batch_obs = torch.FloatTensor(batch_obs).to(self.device)
        if self.share_policy:
            batch_obs = batch_obs.view(-1, self.num_agents, self.dim_obs)
            if is_target:   # Target q networks
                q_vals = self.target_agent_q_nets[0](batch_obs)
            else:   # Action q networks
                q_vals = self.agent_q_nets[0](batch_obs)

            if action is not None:  # return q value of state-action pair
                action_batch = action.view(-1, self.num_agents, 1)
                q_vals = torch.gather(q_vals, 2, action_batch)
        else:
            batch_obs = batch_obs.view(-1, self.num_agents, self.dim_obs)    # [batch_size, num_agents, dim_obs]
            if is_target:
                q_vals = torch.stack([self.target_agent_q_nets[i](batch_obs[:,i,:]) for i in range(self.num_agents)],
                                     axis=1)
            else:
                q_vals = torch.stack([self.agent_q_nets[i](batch_obs[:,i,:]) for i in range(self.num_agents)],
                                     axis=1)

            if action is not None:
                action_batch = action.view(-1, self.num_agents, 1)
                q_vals = torch.gather(q_vals, dim=2, index=action_batch).view(-1, self.num_agents)
        return q_vals

    def append_transition(self, obs, action, reward, obs_):
        """The buffer has num_agents*buffer_size: the consecutive 3 is a three experience"""
        assert self.num_agents == len(obs), "In append_transition func, the num_agents != len(obs)"
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        # Buffer: [obs (num_agents * dim_obs), action (3), reward (3), obs_ (num_agents * dim_obs) 114 in total
        self.buffer[self.buffer_ptr] = np.concatenate((np.concatenate([obs[i] for i in range(len(obs))]),
                                                       action,
                                                       reward,
                                                       np.concatenate([obs_[i] for i in range(len(obs_))])))
        self.buffer_ptr += 1

    def hard_target_update(self):
        print("Hard update targets")
        for i in range(len(self.agent_q_nets)):
            self.target_agent_q_nets[i].load_state_dict(self.agent_q_nets[i].state_dict())

    def prep_train(self):
        """Used to net.train()"""
        for agent in self.agent_q_nets:
            agent.train()

    def prep_eval(self):
        """Used to net.eval()"""
        for agent in self.agent_q_nets:
            agent.eval()

    def save_network(self, iter_cnt):
        """Save network to test directory"""
        for idx, agent in enumerate(self.agent_q_nets):
            torch.save(agent.state_dict(), self.save_dir+"\\net_param_"+str(idx)+"_iter_"+str(iter_cnt)+".pkl")

    def restore(self):
        for idx, agent in enumerate(self.agent_q_nets):
            path = self.save_dir+"\\net_param_"+str(idx)+".pkl"
            agent.load_state_dict(torch.load(path))
