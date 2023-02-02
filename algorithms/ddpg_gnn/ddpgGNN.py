from algorithms.base_agent import BaseAgent
from algorithms.ddpg_gnn.policy import PolicyNet
from algorithms.ddpg_gnn.value_net import ValueNet

import numpy as np
import torch
# from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.functional as F

class ddpg_GNN_Agent(BaseAgent):
    """The training part largely refer to 
    https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py
    """
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus
        self.dim_actions= env_config["num_nodes"]        # Action vector's dimension
        self.norm_factor = [100.0, 100.0, 100.0, 5000.0, 100.0]
        self.len_vec = np.zeros(self.edge_index.shape[1])
        for i in range(self.len_vec.shape[0]):
            self.len_vec[i] = self.len_mat[ self.edge_index[0,i], self.edge_index[1,i] ]

        # Hyper parameters
        self.lr = args.lr
        self.tau = args.tau
        self.discount = args.gamma

        # Initialize buffer, a transition would be [s,a,r,d,s'], where d is don
        # action: [n_node*1]; state: dim_obs; reward, done, time_step: scalar
        self.buffer = np.zeros([self.buffer_size, self.dim_obs*2+self.num_nodes+2])
        self.buffer_ptr = 0
        self.buffer_high = 0

        # Initialize policy network
        policy_network_config = {
            "n_node": self.num_nodes, 
            "out_channels": 5,      # output of GNN: an encoded information, derived from neighbour nodes and node itself. 
            "dim_node_obs": 3,       # node features: idle drivers, demands, upcoming cars.  
            "dim_edge_obs": 2,     # edge features: edge_traffic, length
            "dim_message": 8,       # Dimension of message from neighbour node in MPNN
            "dim_update": 8,       # Dimension of update fianl output
            "message_hidden_layer": [128, 32],     # Dimension of hidden layer in message network in MPNN (Message Passing Neural Network)
            "update_hidden_layer": [128, 32],      # Dimension of hidden layer in update network in MPNN
            "output_hidden_layer": [128, 64],       # Dimension of hidden layer in the mlp after GNN
            "device": self.device,
        }
        self.actor = PolicyNet(policy_network_config, max_bonus=self.max_bonus, min_bonus=self.min_bonus)

        # Initialize the value network
        value_network_config = {
            "n_node": self.num_nodes, 
            "out_channels": 1,      # output of GNN: an encoded information, derived from neighbour nodes and node itself. 
            "dim_node_obs": 4,       # node features: idle drivers, demands, upcoming cars.  
            "dim_edge_obs": 2,     # edge features: edge_traffic, length
            "dim_message": 8,       # Dimension of message from neighbour node in MPNN
            "dim_update": 8,       # Dimension of update fianl output
            "message_hidden_layer": [128, 64],     # Dimension of hidden layer in message network in MPNN (Message Passing Neural Network)
            "update_hidden_layer": [128, 64],      # Dimension of hidden layer in update network in MPNN
            "output_hidden_layer": [128, 64],        # Dimension of hidden layer in the mlp after GNN
            "device": self.device
        }
        self.critic = ValueNet(value_network_config)

        # Initialize network
        self.init_network()

        # Copy target network
        self.actor_target = PolicyNet(policy_network_config, max_bonus=self.dim_action, min_bonus=self.min_bonus)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_target = ValueNet(value_network_config)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5*self.lr)

        # Randomness parameters
        self.epsilon_max = args.max_epsilon
        self.epsilon_min = args.min_epsilon
        self.depsilon = (self.epsilon_max - self.epsilon_min) / args.decre_epsilon_episodes
        self.epsilon = self.epsilon_max
        self.is_training = True

    def choose_action(self, obs, is_random=False):
        if is_random:
            actions = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            x, e, i = self.obs2data(obs)
            with torch.no_grad():
                actions = self.actor(x, e, i).detach().cpu().numpy()[0]
            # add gaussan noise for exploration
            actions = ( actions + np.random.normal(0, self.epsilon, self.dim_actions) ).clip(self.min_bonus, self.max_bonus)    
            self.epsilon = np.maximum(self.epsilon-self.depsilon, self.epsilon_min) 
        return actions
        # [0] is because of actor would return a list of actions, when only one obs is inputed, then it would output a list of only one element

    def append_transition(self, obs, action, reward, done, obs_, info):
        """Store transition"""
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate((self.obs2array(obs), action, [reward], [done], self.obs2array(obs_)))
        self.buffer_ptr += 1
        self.buffer_high = np.minimum( self.buffer_high+1, self.buffer_size )

    def learn(self):
        """
        DDPG learn
        """        
        if self.buffer_high >= self.batch_size:
            self.prep_train()
            # sample transitions
            sample_index = np.random.choice(self.buffer_high, self.batch_size, replace=False)
            batch_memory = self.buffer[sample_index, :]
            batch_state = batch_memory[:, :self.dim_obs]
            batch_action = torch.LongTensor(
                batch_memory[:, self.dim_obs: self.dim_obs+self.num_nodes]).to(self.device)
            batch_reward = torch.FloatTensor(
                batch_memory[:, self.dim_obs+self.num_nodes: self.dim_obs+self.num_nodes+1]).to(self.device)
            batch_done = torch.LongTensor(1-batch_memory[:, -self.dim_obs-1:-self.dim_obs]).to(self.device)   
            batch_next_state = batch_memory[:, -self.dim_obs:]
        else:
            print("Transition number is lesser than batch_size, continue training. ")
            return

        # Current q estimate
        x, e, i = self.obs2data(batch_state, batch_action)  # node_features, edge_features, edge_index
        curr_q_values = self.critic(x, e, i)
        
        # Target q value
        x, e, i = self.obs2data(batch_next_state)
        next_action = self.actor_target(x, e, i)
        x, e, i = self.obs2data(batch_next_state, next_action)
        target_q = self.critic_target(x, e, i)
        target_q = batch_reward + (batch_done*target_q).detach()
        # print("Current q: ", curr_q_values)
        # print("Target q: ", target_q )

        # Compute critic loss
        critic_loss = F.mse_loss(curr_q_values, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        a_x, a_e, a_i = self.obs2data(batch_state)
        action = self.actor(a_x, a_e, a_i)
        c_x, c_e, c_i = self.obs2data(batch_state, action)
        actor_loss = -self.critic(c_x, c_e, c_i).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Record the loss trajectory
        critic_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.critic.parameters()])
        self.writer.add_scalar("critic_grad_max", np.max(critic_grad), self.train_steps)
        actor_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.actor.parameters()])
        self.writer.add_scalar("actor_grad_max", np.max(actor_grad), self.train_steps)
        self.writer.add_scalar("critic_loss", critic_loss.item(), self.train_steps)
        self.writer.add_scalar("actor_loss", actor_loss.item(), self.train_steps)
        self.train_steps += 1

        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def obs2array(self, obs):
        """This function transform state in dict form to numpy array (n_agents*dim_node_obs+n_edges*dim_edge_obs)"""
        if type(obs)==dict: # When obs is just one state
            state = np.concatenate([obs["idle_drivers"]/self.norm_factor[0], 
                                    obs["upcoming_cars"]/self.norm_factor[1], 
                                    obs["demands"]/self.norm_factor[2], 
                                    np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])])/self.norm_factor[3], # Normalizae edge_traffic
                                    self.len_vec/self.norm_factor[4]])
        else:   # When obs is a batch of states
            state = np.vstack([np.concatenate([obs[i]["idle_drivers"]/self.norm_factor[0], 
                                               obs[i]["upcoming_cars"]/self.norm_factor[1], 
                                               obs[i]["demands"]/self.norm_factor[2], 
                                               np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])])/self.norm_factor[3], # Normalizae edge_traffic
                                               self.len_vec/self.norm_factor[4]]) for i in range(len(obs))])
        return state

    def obs2data(self, obs, action=None):
        """This function convert obs(dict) to network's input data(torch_geometric.Data)
        @params: 
            obs: observation (dict) check self.obs2array() to find out its contents. 
        @return:
            data: distributed agents' obervations. In torch_geometric.Data form
        """
        if type(obs) == dict:   # Converge a single dict-type observation to a torch_geometric.data.Data type variable
            x = torch.FloatTensor(np.vstack([obs["idle_drivers"]/self.norm_factor[0], obs["upcoming_cars"]/self.norm_factor[1], obs["demands"]/self.norm_factor[2]]).T)
            traffic = np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])])/self.norm_factor[3]    # Normalize traffic
            edge_attr = torch.FloatTensor(np.vstack([traffic, self.len_vec/self.norm_factor[4]]).T)
            return x.to(self.device), edge_attr.to(self.device), torch.LongTensor(self.edge_index).to(self.device)
        elif type(obs)==np.ndarray: # Load from buffer
            if len(obs.shape)==1:   # obs contains a single observation
                raise NotImplementedError("training batch data should not be 0, plesae change batch size to more than 1")
            else:   # obs is a batch of observation
                obs = torch.FloatTensor(obs).to(self.device)
                batch_size = obs.shape[0]
                if action is not None:    # Value network input
                    x = torch.hstack([ obs[:, :self.dim_node_obs*self.num_nodes], action ]).\
                                reshape(-1, self.dim_node_obs+1, self.num_nodes).transpose(1,2).reshape(-1, self.dim_node_obs+1)
                else:   # Actor network input
                    x = obs[:, :self.dim_node_obs*self.num_nodes].\
                        reshape(-1, self.dim_node_obs, self.num_nodes).transpose(1,2).reshape(-1, self.dim_node_obs)

                node_state_end_idx = self.dim_node_obs*self.num_nodes
                edge_attr = torch.cat([state[node_state_end_idx : ].reshape(-1, self.dim_edge_obs, self.edge_index.shape[1]).transpose(1,2)\
                                        for state in obs]).reshape(-1, self.dim_edge_obs).to(self.device)
                data_edge_index = torch.cat([torch.LongTensor(self.edge_index+i*self.num_nodes)\
                                                   for i in range(batch_size)], dim=1).reshape(self.dim_edge_obs,-1)
        return x.to(self.device), edge_attr.to(self.device), data_edge_index.to(self.device)

    def init_network(self):
        for name, param in self.actor.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.kaiming_normal_(param)
        for name, param in self.critic.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.kaiming_normal_(param)

    def prep_train(self):
        self.actor.train()
        self.critic.train()
    
    def save_network(self):
        """Heuristic do not need save. """
        pass