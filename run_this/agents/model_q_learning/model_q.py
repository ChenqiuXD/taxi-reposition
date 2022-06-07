# Implementation of DQN
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from rl_algo.utils.base_agent import BaseAgent
from transition_model.train_model.model import Net
from torch_geometric.data import Data

class M_QAgent(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        
        self.edge_index = env_config["edge_index"]
        self.len_mat = env_config["len_mat"]
        self.share_policy = True
        print("Using single agent model-based q-learning, sharing policy is true.")
        self.q_net = QNetwork(n_features=self.dim_obs, n_actions=self.dim_action, n_agents=self.num_agents)
        self.target_q_net = copy.deepcopy(self.q_net)

        # Buffer
        self.buffer_size = self.args.buffer_size
        self.buffer_ptr = 0
        # a transition is [s,a,r,s'], action is a [num_agent*1] vector, plus reward and time_step NOTE THAT HERE WE ADD TIME_STEP FOR TRANSITION PREDICTION
        # !!! Please note that time_step is not very general in traditional rl 
        self.buffer = np.zeros([self.buffer_size, self.dim_obs*2+self.num_agents+2])    # a transition is [s, a, r, s'], action is a [num_agent*1] vector

        # Optimizer
        self.optim = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        self.loss_fn = nn.MSELoss()

        # Model relevant parameters
        self.model_path = os.getcwd()+"\\transition_model\\train_model"
        agent_q_config = {
            "num_node": 5,
            "dim_action": 5,
            "dim_node_obs": 3,
            "dim_edge_obs": 2,
            "dim_message": 1,
            "out_channels": 5,
            "message_hidden_layer": 128,
            "update_hidden_layer": 128,
        }
        self.model = self.load_transition_model(agent_q_config)
        try:
            self.num_gen_sample = self.args.generate_sample    # How many samples would be generated at each iteration. default: 1
        except:
            raise RuntimeError("Model-based RL should have argument generate_samples, yet it's not found in args. ")
        self.time_mats = np.zeros([self.episode_length, self.num_node, self.num_node])

        self.save_dir = os.path.abspath('.')+"\\run_this\\agents\\q_learning\\"    # Untested, please test it before using

    def learn(self):
        """Use samples from replay buffer to update q-network"""
        # Determine the number of generated data
        sample_batch_size = int(self.batch_size/(1+self.num_gen_sample))  # Data sampled from self.buffer
        
        # sample batch from memory
        sample_index = np.random.choice(self.buffer_size, self.batch_size)
        batch_memory = self.buffer[sample_index, :]
        batch_state = batch_memory[:, :self.dim_obs]
        batch_action = torch.LongTensor(batch_memory[:, self.dim_obs:
                                        self.dim_obs+self.num_agents].astype(int))
        batch_reward = batch_memory[:, self.dim_obs+self.num_agents:
                                    self.dim_obs+self.num_agents+1]
        batch_next_state = batch_memory[:, -self.dim_obs-1:-1]
        # The last element is "time_step", therefore the "-1" in above line. 

        # Generate data from model
        # Generate model_based data
        gen_batch_reward, gen_next_obs = self.generate_data(sample_index[sample_batch_size:])
        batch_reward[sample_batch_size:] = gen_batch_reward.reshape([len(sample_index)-sample_batch_size, 1])
        batch_next_state[sample_batch_size:] = gen_next_obs

        # Convert data to torch required data form
        # TODO: Please test this. 
        batch_state = self.obs2data(batch_state).to(self.device)
        batch_next_state = self.obs2data(batch_next_state).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device)

        # q_eval
        batch_action_encoded = torch.zeros(batch_reward.shape[0])
        # Encode actions ([num_agents*1]) to index (scalar, range from 0-pow(num_agents, dim_actions))
        for i in range(self.num_agents):
            batch_action_encoded += batch_action[:,i] * pow(self.num_agents, i)
        q_eval = self.q_net(batch_state).gather(1, batch_action_encoded.long().view(-1,1))
        q_next = self.target_q_net(batch_next_state).detach()
        # q_next = self.q_network(batch_next_state)
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_fn(q_eval, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def generate_data(self, data_idx):
        """This function generate data based on self.model which outputs the equlibrium actions of node
        @params: 
            data_idx: (list) a list of sampled data_idx which would be used to predict the transition using model
        @returns:
            batch_next_state: (ndarray, [self.dim_overall_obs,len(data_idx)]) model prediction of next state
            overall_cost: (FloatTensor, [1*len(data_idx)]) calculated cost based on model prediction
        """
        batch_data = self.buffer[data_idx]
        gen_batch_size = len(data_idx)
        # if gen_batch_size == 1: # Convert a single data to list. Since we assume batch_data>=1
        #     batch_data = [batch_data]

        # Convert observation and action to GNN input (cannot use obs2data since there includes action)
        obs = batch_data[:, :self.dim_overall_obs]
        batch_action = batch_data[:, self.dim_overall_obs : self.dim_overall_obs+self.num_agents]
        x = torch.FloatTensor(np.concatenate(np.concatenate(
                [obs[:, :self.dim_node_obs*self.num_agents], batch_action], axis=1).\
                reshape(-1, self.dim_node_obs+1, self.num_agents).swapaxes(1,2)))
        node_state_end_idx = self.dim_node_obs*self.num_agents
        edge_attr = torch.FloatTensor(np.concatenate(
                        obs[:, node_state_end_idx : ].\
                        reshape(-1, self.dim_edge_obs, self.edge_index.shape[1]).swapaxes(1,2)))
        data_edge_index = np.concatenate([[self.edge_index+i*self.num_agents]\
                                            for i in range(gen_batch_size)]).swapaxes(0,1).reshape(self.dim_edge_obs,-1)
        gen_data = Data(x=x, edge_attr=edge_attr, edge_index=torch.LongTensor(data_edge_index))

        nodes_action_prob = self.model(gen_data).detach().cpu().numpy().reshape(-1, self.num_agents, self.num_agents)
        idle_drivers = obs[:, :self.num_agents] # idle_drivers at last step
        nodes_action = np.vstack([ (nodes_action_prob[i].T*idle_drivers[i]).T for i in range(nodes_action_prob.shape[0]) ]).reshape(-1, self.num_agents, self.num_agents)

        # Calc cost
        node_all_cars = np.sum(nodes_action, axis=1) + obs[:, self.num_agents: self.num_agents*2]    # re-positioned idle_drivers plus upcoming cars
        nodes_distribution = np.vstack([node_all_cars[i] / np.sum(node_all_cars,axis=1)[i] for i in range(node_all_cars.shape[0])])
        batch_demand = obs[:, self.num_agents*2:self.num_agents*3]
        demand_distribution = np.vstack([ batch_demand[i] / np.sum(batch_demand, axis=1)[i] for i in range(batch_demand.shape[0]) ])

        # mse loss of two distribution
        idle_cost = np.sqrt(np.sum((nodes_distribution-demand_distribution)**2, axis=1))*3

        # Calculate the travelling time
        batch_time_step = batch_data[:, -1].astype(int)  # The last element in buffer is "time_step"
        max_time = np.array([np.max(self.time_mats[batch_time_step[i]]) for i in range(batch_data.shape[0])])
        avg_travelling_cost = np.array([np.sum((nodes_action*self.time_mats[batch_time_step])[i]) for i in range(nodes_action.shape[0])])\
                              /(np.array([ np.sum(nodes_action[i]*max_time[i]) for i in range(nodes_action.shape[0]) ]))
        
        # Calculate the bonus
        bonuses_cost = np.array([np.sum(nodes_action[i]*batch_action[i]) for i in range(nodes_action.shape[0])])/\
                       np.array([np.sum(nodes_action[i]) for i in range(nodes_action.shape[0])])*self.dim_action    # dim_action=max_bonus=5
        overall_cost = (0.4*idle_cost+0.4*avg_travelling_cost+0.2*bonuses_cost)*100

        # Compute next state (only changes idle_drivers of obs_ since upcoming_cars, demands, traffic remains the same)
        idle_drivers = np.maximum(0, np.array([np.sum(nodes_action[i], axis=0) for i in range(nodes_action.shape[0])])   # re-positioned drivers
                                     +batch_data[:, self.num_agents: self.num_agents*2]     # plus upcoming cars
                                     -batch_data[:, self.num_agents*2:self.num_agents*3]).astype(int)   # minus demands
        batch_next_state = self.buffer[data_idx, -self.dim_overall_obs-1:-1]
        batch_next_state[:, :self.num_agents] = idle_drivers

        return overall_cost, batch_next_state

    def append_transition(self, obs, action, reward, obs_, time_mat):
        """Store transition"""
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        self.time_mats[obs["time_step"]] = time_mat

        self.buffer[self.buffer_ptr] = np.concatenate((self.to_state(obs), action, [reward], self.to_state(obs_), [obs["time_step"]]))
        self.buffer_ptr += 1

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

    def choose_action(self, obs, is_random=False):
        """Choose actions based on q-network"""
        actions = np.zeros(self.num_agents)
        if np.random.uniform()>=self.e_greedy or is_random:
            for i in range(self.num_agents):
                actions[i] = np.random.choice(np.arange(self.dim_action))
        else:
            action_value = self.q_net(torch.FloatTensor(self.to_state(obs))).detach().numpy()
            action_list = np.where(action_value == np.max(action_value))[0]
            action_encoded = np.random.choice(action_list)
            for i in range(self.num_agents):
                actions[i] = action_encoded % self.num_agents
                action_encoded = (action_encoded-actions[i])/self.num_agents
        return actions

    def prep_train(self):
        self.q_net.train()
    
    def prep_eval(self):
        self.q_net.eval()

    def hard_target_update(self):
        print("Hard update targets")
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def load_transition_model(self, agent_q_config):
        """Load transition model learned from transition model step (a gnn used to approximate the reaction of drivers in equlibrium)"""
        # Instantiate a model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = {
            # For making decision, the obs is 3. When predicting transition, we add actions, therefore the in_channels  shoudl plus 1
            "in_channels": agent_q_config["dim_node_obs"]+1,
            "edge_channels": agent_q_config["dim_edge_obs"], 
            "out_channels": agent_q_config["out_channels"], 
            "dim_message": 1,
            "message_hidden_layer": [32, 32],
            "update_hidden_layer": [32, 32],
        }
        model = Net(agent_q_config["num_node"], config, device, flow='source_to_target').to(device)

        # Load parameters
        model.load_state_dict(torch.load(self.model_path+"\\tran_model.pkl"))
        return model

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
    def __init__(self, n_features, n_actions, n_agents, l1=256, l2=128):
        super(QNetwork, self).__init__()
        # Construct network - three layers
        dim_agents_actions = pow(n_agents, n_actions)
        self.fc1 = nn.Linear(n_features, max(l1, 4*dim_agents_actions))    # input is [n_features + agents' action (1 for each agents)]
        self.fc1.weight.data.normal_(0, 1e-4)
        self.fc2 = nn.Linear(max(l1, 4*dim_agents_actions), max(l2, 4*dim_agents_actions))
        self.fc2.weight.data.normal_(0, 1e-4)
        self.fc3 = nn.Linear(max(l2, 4*dim_agents_actions), pow(n_agents, n_actions))     # output Q(s,a) a\in R^{n_agents}
        self.fc3.weight.data.normal_(0, 1e-4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x