import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class QMix(nn.Module):
    """Q-Mix neural network containing a hyper-network and several individual agent networks"""
    def __init__(self, args, mixer_config):
        super(QMix, self).__init__()
        self.args = args
        self.device = self.args.device
        self.num_agent = mixer_config['num_agent']
        self.dim_cent_obs = mixer_config['dim_cent_obs']
        self.num_mixer_q_inps = self.num_agent

        # Layer parameters
        self.hidden_layer_dim = mixer_config['hidden_layer_dim']
        self.hypernet_hidden_dim = mixer_config['hypernet_hidden_dim']
        self.hypernet_layers = mixer_config['hypernet_layers']

        # hypernets output the weight and bias for the 2 layer MLP which takes in state and agent Qs and outputs Q_tot
        if self.hypernet_layers == 2:
            # each hypernet only has 2 layer to output the weights
            # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
            self.hyper_w1 = nn.Sequential(nn.Linear(self.dim_cent_obs, self.hypernet_hidden_dim), 
                                          nn.ReLU(), 
                                          nn.Linear(self.hypernet_hidden_dim, self.num_agent*self.hidden_layer_dim)).to(self.device)
            self.hyper_w2 = nn.Sequential(nn.Linear(self.dim_cent_obs, self.hypernet_hidden_dim), 
                                          nn.ReLU(), 
                                          nn.Linear(self.hypernet_hidden_dim, self.hidden_layer_dim)).to(self.device)
        elif self.hypernet_layers == 1:
            # 1 layer hypernets: output dimensions are same as above case
            self.hyper_w1 = nn.Linear(self.dim_cent_obs, self.num_agent*self.hidden_layer_dim).to(self.device)
            self.hyper_w2 = nn.Linear(self.dim_cent_obs, self.hidden_layer_dim).to(self.device)

        # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
        self.hyper_b1 = nn.Linear(self.dim_cent_obs, self.hidden_layer_dim).to(self.device)
        # hyper_b2 outptus bias vector of dimension (1 x 1)
        self.hyper_b2 = nn.Linear(self.dim_cent_obs, 1).to(self.device)

    def forward(self, agent_q_inps, states):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) cent_obs_state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        # agent_q_inps = agent_q_inps.to(self.device)
        # states = states.to(self.device)

        batch_size = agent_q_inps.size(0)
        # states = states.view(-1, self.dim_cent_obs).float()
        # reshape agent_q_inps into shape (batch_size x 1 x N) to work with torch.bmm
        agent_q_inps = agent_q_inps.view(-1, 1, self.num_agent).float()

        # get the first layer weight matrix batch, apply abs val to ensure nonnegative derivative
        w1 = torch.abs(self.hyper_w1(states))
        # get first bias vector
        b1 = self.hyper_b1(states)
        # reshape to batch_size x N x Hidden Layer Dim (there's a different weight mat for each batch element)
        w1 = w1.view(-1, self.num_agent, self.hidden_layer_dim)
        # reshape to batch_size x 1 x Hidden Layer Dim
        b1 = b1.view(-1, 1, self.hidden_layer_dim)
        # pass the agent qs through first layer defined by the weight matrices, and apply Elu activation
        hidden_layer = F.elu(torch.bmm(agent_q_inps, w1) + b1)

        # get second layer weight matrix batch
        w2 = torch.abs(self.hyper_w2(states))
        # get second layer bias batch
        b2 = self.hyper_b2(states)
        # reshape to shape (batch_size x hidden_layer dim x 1)
        w2 = w2.view(-1, self.hidden_layer_dim, 1)
        # reshape to shape (batch_size x 1 x 1)
        b2 = b2.view(-1, 1, 1)
        # pass the hidden layer results through output layer, with no activataion
        out = torch.bmm(hidden_layer, w2) + b2
        # reshape to (batch_size, 1, 1)
        q_tot = out.view(batch_size, -1, 1)

        return q_tot
