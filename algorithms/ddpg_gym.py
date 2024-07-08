# reference: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py 
# and https://github.com/Jacklinkk/Graph_CAVs
import os
import pickle
import copy
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.utils.replay_buffer import ReplayBuffer
from algorithms.utils.prioritized_replay_buffer import PrioritizedReplayBuffer

criterion = nn.MSELoss()

class DDPG():
    def __init__(self, dim_state, dim_action, lr=1e-3):
        super().__init__()

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        # Construct network
        self.actor = NonGraph_Actor_Model(dim_state, dim_action).to(self.device)
        self.critic = NonGraph_Critic_Model(dim_state, dim_action).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # Noisy exploration
        self.explore_noise = OUActionNoise(mu=np.array([0]))

        # Hyper-parameters
        self.buffer = ReplayBuffer(size=10**6)
        self.batch_size = 64
        self.tau = 5e-3
        self.gamma = 0.9

        # Utils
        self.time_counter = 0

    def choose_action(self, s, is_random=False):
        if is_random:
            action = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            s = torch.FloatTensor(s).to(self.device)
            action = self.actor(s)
            noise = torch.as_tensor(self.explore_noise(), dtype=torch.float32).to(self.device)
            action = action + noise
            action = torch.clamp(action, -1, +1)

        return action.detach().cpu().numpy()

    def append_transition(self, s, a, r, d, s_, info=None):
        # Append transition
        self.buffer.add(s, a, r, s_, d)
    
    def sample_memory(self):
        # Call the sampling function in replay_buffer
        data_sample = self.buffer.sample(self.batch_size)
        return data_sample

    def loss_process(self, loss, weight):
        # Calculation of loss based on weight
        weight = torch.as_tensor(weight, dtype=torch.float32).to(self.device)
        loss = torch.mean(loss * weight.detach())

        return loss

    def learn(self):
        # ------whether to return------ #
        self.time_counter += 1
        if (self.time_counter <= 2 * self.batch_size):
            return

        self.prep_train()

        # ------ calculates the loss ------ #
        # Experience pool sampling, samples include weights and indexes,
        # data_sample is the specific sampled data
        info_batch, data_batch = self.sample_memory()
        
        # Initialize the loss matrix
        actor_loss = []
        critic_loss = []

        # Extract data from each sample in the order in which it is stored
        # ------loss of critic network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            # target value
            with torch.no_grad():
                action_target = self.actor_target(next_state)
                critic_value_next = self.critic_target(next_state, action_target).detach()
                critic_target = reward + self.gamma * critic_value_next * (1 - done)
            critic_value = self.critic(state, action)

            critic_loss_sample = F.smooth_l1_loss(critic_value, critic_target)
            critic_loss.append(critic_loss_sample)

        # critic network update
        critic_loss_e = torch.stack(critic_loss)
        # critic_loss_total = self.loss_process(critic_loss_e, info_batch['weights'])
        critic_loss_total = critic_loss_e.mean()
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # ------loss of actor network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)

            mu = self.actor(state)
            actor_loss_sample = -1 * self.critic(state, mu)
            actor_loss_s = actor_loss_sample.mean()
            actor_loss.append(actor_loss_s)

        # actor network update
        actor_loss_e = torch.stack(actor_loss)
        # actor_loss_total = self.loss_process(actor_loss_e, info_batch['weights'])
        actor_loss_total = actor_loss_e.mean()
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()
    
        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def prep_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s) 

    def save_model(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pkl")
        torch.save(self.critic.state_dict(), filename + "_critic.pkl")   
    
    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pkl"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pkl"))

# Defining Ornstein-Uhlenbeck noise for stochastic exploration processes
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OUActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class NonGraph_Actor_Model(nn.Module):
    """
       1.N is the number of nodes
       2.F is the feature length of each nodes
       3.A is the dimension of action
    """
    def __init__(self, dim_state, dim_action):
        super(NonGraph_Actor_Model, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action

        # Encoder
        self.policy_1 = nn.Linear(self.dim_state, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Actor network
        self.pi = nn.Linear(32, self.dim_action)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):
        """
            1.Observation is the state observation matrix.
        """

        # X_in, _, RL_indice = datatype_transmission(observation, self.device)
        X_in = observation

        # Policy
        X_policy = self.policy_1(X_in)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Pi
        pi = self.pi(X_policy)
        action = torch.tanh(pi)

        return action

class NonGraph_Critic_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, dim_state, dim_action):
        super(NonGraph_Critic_Model, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action

        # Policy network
        self.policy_1 = nn.Linear(self.dim_state+self.dim_action, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Critic network
        self.value = nn.Linear(32, 1)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, action):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, and RL_indice.
            3.X_in is the node feature matrix, RL_indice is the reinforcement learning
            index of controlled vehicles.
        """

        # X_in, _, RL_indice = datatype_transmission(observation, self.device)
        X_in = observation

        # Policy
        X_in = torch.cat((X_in, action))
        X_policy = self.policy_1(X_in)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Value
        V = self.value(X_policy)

        return V
