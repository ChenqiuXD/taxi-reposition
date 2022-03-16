import copy
import torch.nn as nn
import torch.optim

from QMix.q_mix.q_mix import QMix
from utils.agent_q_function import AgentQFunction
from VDN.vdn import VDN
import numpy as np
import os


class QMixer(VDN):
    """
    Trainer class for QMix with MLP policies.
    """
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

        # Initialize mixer network
        mixer_config = {
            'num_agent': self.num_agents,
            'dim_cent_obs': self.dim_obs * self.num_agents,  # central observation (combine agents' paritial obs)
            'hidden_layer_dim': 32,
            'hypernet_hidden_dim': 128,
            'hypernet_layers': 1
        }
        self.mixer = QMix(self.args, mixer_config).to(self.device)

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
        self.target_agent_q_nets = copy.deepcopy(self.agent_q_nets)
        self.target_mixer_net = copy.deepcopy(self.mixer)

        # Collect all parameters
        self.parameters = []
        for agent in self.agent_q_nets:
            self.parameters += agent.parameters()
        self.parameters += self.mixer.parameters()

        # self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        # self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr)
        self.optimizer = torch.optim.RMSprop(self.parameters, lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.save_dir = os.path.abspath('.') + "\\QMix"

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
            curr_mixed_q = self.mixer(curr_agent_q_vals, batch_state).view(-1, 1)

            # Next state Q values
            next_agent_q_vals_all = self.get_q_value(batch_next_state, is_target=True)
            next_agent_q_vals, _ = torch.max(next_agent_q_vals_all, dim=2)
            next_mixed_q = self.target_mixer_net(next_agent_q_vals, batch_next_state).view(-1, 1)

            # Compute Bellman loss
            target_mixed_q = (batch_reward + self.gamma * next_mixed_q).detach()
            loss = self.loss_func(curr_mixed_q.squeeze(), target_mixed_q.squeeze())
            # print("Current loss is: ", loss)

            # optimise
            self.optimizer.zero_grad()
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
            self.optimizer.step()
        else:
            raise NotImplementedError("Not sharing policy is not implemented for qmix methods")

    def hard_target_update(self):
        print("Hard update targets")
        for i in range(len(self.agent_q_nets)):
            self.target_agent_q_nets[i].load_state_dict(self.agent_q_nets[i].state_dict())
        self.target_mixer_net.load_state_dict(self.mixer.state_dict())

    def prep_train(self):
        """Used to net.train()"""
        self.mixer.train()
        for agent in self.agent_q_nets:
            agent.train()

    def prep_eval(self):
        """Used to net.eval()"""
        self.mixer.eval()
        for agent in self.agent_q_nets:
            agent.eval()

    def save_network(self, iter_cnt):
        """Save network to test directory"""
        torch.save(self.mixer.state_dict(), self.save_dir+"\\mixer_net_params"+str(iter_cnt)+".pkl")
        for idx, agent in enumerate(self.agent_q_nets):
            torch.save(agent.state_dict(), self.save_dir+"\\net_param_"+str(idx) + "_iter_" + str(iter_cnt) + ".pkl")

    def restore(self):
        path = self.save_dir+"\\mixer_net_params.pkl"
        self.mixer.load_state_dict(torch.load(path))
        for idx, agent in enumerate(self.agent_q_nets):
            path = self.save_dir+"\\net_param_"+str(idx)+".pkl"
            agent.load_state_dict(torch.load(path))
