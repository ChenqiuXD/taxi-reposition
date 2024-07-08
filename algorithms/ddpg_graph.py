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

from algorithms.base_agent import BaseAgent
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

from algorithms.utils.replay_buffer import ReplayBuffer
from algorithms.utils.prioritized_replay_buffer import PrioritizedReplayBuffer

criterion = nn.MSELoss()
NORMAALIZATION_FACTOR = 100.0

class DDPG_graph(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        else:
            self.use_cuda = False
        if args.seed > 0:
            self.seed(args.seed)

        self.device = args.device
        self.dim_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.dim_actions= env_config["num_nodes"]        # Action vector's dimension
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        # Construct graph network
        edge_index = env_config["edge_index"]
        dim_features = int(self.dim_states / env_config["num_nodes"])
        dim_actions = int(self.dim_actions / env_config["num_nodes"])
        self.actor = Graph_Actor_Model(env_config["num_nodes"], dim_features, dim_actions, self.min_bonus, self.max_bonus, edge_index).to(self.device)
        self.critic = Graph_Critic_Model(env_config["num_nodes"], dim_features, dim_actions, self.min_bonus, self.max_bonus, edge_index).to(self.device)
    
        # Target networks
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Noisy exploration
        self.explore_noise = OUActionNoise(mu=np.zeros([self.dim_action]))

        # Hyper-parameters
        self.buffer = ReplayBuffer(size=args.buffer_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.discount = args.gamma

        # Utils
        self.time_counter = 0
        self.loss_record = collections.deque(maxlen=100)
        self.is_training = True

    def choose_action(self, s, is_random=False):
        if is_random:
            action = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            state = torch.FloatTensor(self.s2obs(s)).to(self.device)

            # Generate action
            action = self.actor(state)
            noise = torch.as_tensor(self.explore_noise(), dtype=torch.float32).to(self.device)
            action = action + noise
            action = torch.clamp(action, self.min_bonus, self.max_bonus)

        return action.detach().numpy()

    def append_transition(self, s, a, r, d, s_, info):
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
            state = self.s2obs(state)
            next_state = self.s2obs(next_state)
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
        critic_loss_total = self.loss_process(critic_loss_e, info_batch['weights'])
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # ------loss of actor network------ #
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            state = self.s2obs(state)
            next_state = self.s2obs(next_state)

            mu = self.actor(state)
            actor_loss_sample = -1 * self.critic(state, mu)
            actor_loss_s = actor_loss_sample.mean()
            actor_loss.append(actor_loss_s)

        # actor network update
        actor_loss_e = torch.stack(actor_loss)
        actor_loss_total = self.loss_process(actor_loss_e, info_batch['weights'])
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()

        # ------Updating PRE weights------ #
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.update_priority(info_batch['indexes'], (critic_loss_e + actor_loss_e))

        # ------Record loss------ #
        self.loss_record.append(float((critic_loss_total + actor_loss_total).detach().cpu().numpy()))
    
        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.writer.add_scalar("critic_loss", critic_loss_total.item(), self.time_counter)
        self.writer.add_scalar("actor_loss", actor_loss_total.item(), self.time_counter)

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        obs = np.concatenate([s['idle_drivers']/NORMAALIZATION_FACTOR, s['demands']/NORMAALIZATION_FACTOR])
        return torch.FloatTensor(obs).to(self.device)

    def prep_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def reset(self):
        self.random_process.reset_states()

    def load_model(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
                os.path.join(output, "actor.pkl")
        )
        torch.save(
            self.critic.state_dict(),
                os.path.join(output, "critic.pkl")
        )

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)    



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

class Graph_Actor_Model(nn.Module):
    """
        1.N is the number of nodes
        2.F is the feature length of each node
        3.A is the dimension of actions for each node
    """
    def __init__(self, N, F, A, action_min, action_max, edge_index):
        super(Graph_Actor_Model, self).__init__()

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max
        self.edge_index = torch.tensor(edge_index).to(self.device)

        # Encoder
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Actor network
        self.pi = nn.Linear(32, A)

        self.to(self.device)

    def forward(self, observation):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """
        # Reshape observation to match the input shape of self.GraphConv
        # X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)
        X_in = observation.reshape([self.num_agents, -1])

        # Encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        # A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        X_graph = self.GraphConv(X, edge_index=self.edge_index)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Features concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Pi
        pi = self.pi(X_policy)

        # Action limitation
        amplitude = 0.5 * (self.action_max - self.action_min)
        mean = 0.5 * (self.action_max + self.action_min)
        action = amplitude * torch.tanh(pi) + mean

        return action.squeeze()

class Graph_Critic_Model(nn.Module):
    """
        1.N is the number of nodes
        2.F is the feature length of each node
        3.A is the dimension of actions for each node
    """
    def __init__(self, N, F, A, action_min, action_max, edge_index):
        super(Graph_Critic_Model, self).__init__()

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max
        self.edge_index = torch.tensor(edge_index).to(self.device)

        # Encoder
        self.encoder_1 = nn.Linear(F + A, 32)  # Considering action space
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Critic network
        self.value_1 = nn.Linear(32*self.num_agents, 64)
        self.value_2 = nn.Linear(64, 1)

        self.to(self.device)

    def forward(self, observation, action):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """

        # X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)
        X_in = observation.reshape([self.num_agents, -1])

        # Encoder
        X_in = torch.cat((X_in, action.reshape([self.num_agents, -1])), 1)
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        X_graph = self.GraphConv(X, self.edge_index)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy network
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Value calculation
        V = self.value_1(X_policy.flatten())
        V = F.relu(V)
        V = self.value_2(V)

        return V
