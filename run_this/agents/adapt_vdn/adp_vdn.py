import copy
import torch.nn as nn
import torch.optim
from agents.adapt_vdn.agent_q_function import AgentQFunction
from rl_algo.utils.base_agent import BaseAgent
import numpy as np
from torch_geometric.data import Data
from transition_model.train_model.model import Net
import os


class ADP_VDN(BaseAgent):
    """
    Trainer class for QMix with MLP policies.
    """
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

        self.share_policy = self.args.share_policy 
        self.edge_index = env_config["edge_index"]
        self.len_mat = env_config["len_mat"]
        self.dim_node_obs = env_config["dim_node_obs"]
        self.dim_edge_obs = env_config["dim_edge_obs"]
        self.episode_length = args.episode_length
        self.num_node = env_config["num_agents"]
        self.dim_overall_obs = self.dim_obs # including all nodes' features and edges' features

        # Initialize buffer
        self.buffer_size = self.args.buffer_size
        self.buffer_ptr = 0
        # a transition is [s,a,r,s'], action is a [num_agent*1] vector, plus 1 for time_step
        self.buffer = np.zeros((self.buffer_size, 2*self.dim_obs+self.num_agents+2))

        # Initialize model buffer
        self.model_buffer_size = self.args.buffer_size*10
        self.model_buffer_ptr = 0
        self.model_buffer = np.zeros((self.model_buffer_size, 2*self.dim_obs+self.num_agents+2))

        # Initialize agent networks
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
        if self.share_policy:
            self.agent_q_nets = [AgentQFunction(self.args, agent_q_config, id=0).to(self.device)]
        else:
            raise NotImplementedError("Not sharing policy for vdn method is not implemented, please change args.share_policy to true")

        # Initialize by xavier_uniform_
        for agent in self.agent_q_nets:
            for net in agent.modules():
                if isinstance(net, nn.Linear):
                    nn.init.xavier_uniform_(net.weight)
                    nn.init.zeros_(net.bias)

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

        # Load transition_model
        config = {
            # For making decision, the obs is 3. When predicting transition, we add actions, therefore the in_channels  shoudl plus 1
            "in_channels": agent_q_config["dim_node_obs"]+1,
            "edge_channels": agent_q_config["dim_edge_obs"], 
            "out_channels": agent_q_config["out_channels"], 
            "dim_message": 8,
            "message_hidden_layer": [128, 64],
            "update_hidden_layer": [64,32],
        }
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(agent_q_config["num_node"], config, device, flow='source_to_target').to(device)

        # Collect model parameters
        self.model_parameters = Net.parameters()

        self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)
        self.loss_func = nn.MSELoss()
        self.model_optimizer = torch.optim.Adam(params=self.model_parameters, lr=self.lr*2)

        self.save_dir = os.path.abspath('.')+"\\run_this\\agents\\vdn\\"

    def learn(self):
        """Update the network parameters using Q-learning"""
        self.prep_train()
        if self.share_policy:
            sample_index = np.random.choice(self.buffer_size, self.batch_size, replace=False)
            batch_memory = self.buffer[sample_index, :]
            batch_state = self.obs2data(batch_memory[:, :self.dim_obs]).to(self.device)
            batch_action = torch.LongTensor(
                batch_memory[:, self.dim_obs: self.dim_obs+self.num_agents]).to(self.device)
            batch_reward = torch.FloatTensor(
                batch_memory[:, self.dim_obs+self.num_agents: self.dim_obs+self.num_agents+1]).to(self.device)
            batch_next_state = self.obs2data(batch_memory[:, -self.dim_obs:]).to(self.device)

            # Current Q values
            curr_agent_q_vals = self.get_q_value(batch_state, action=batch_action, is_target=False).view(self.batch_size, -1)
            curr_mixed_q = torch.sum(curr_agent_q_vals, dim=1).view(-1, 1)

            # Next state Q values
            next_agent_q_vals_all = self.get_q_value(batch_next_state, is_target=True)
            next_agent_q_vals= torch.max(next_agent_q_vals_all, dim=1)[0].view(self.batch_size, self.num_agents)
            next_mixed_q = torch.sum(next_agent_q_vals, dim=1).view(-1, 1)

            # Compute Bellman loss
            target_mixed_q = (batch_reward + self.gamma * next_mixed_q).detach()
            # target_mixed_q = batch_reward.detach()
            loss = self.loss_func(curr_mixed_q, target_mixed_q)
            print("Current loss is: ", loss)

            # optimise
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError("Not sharing policy for vdn method is not implemented, please change args.share_policy to true")

    def get_q_value(self, batch_obs, action=None, is_target=False):
        """Get the batch q value of state-action pair
        :param: batch_obs is torch_geometric.data.Data type inputs
        :param: action is the batch of actions (dim: [batch_size, num_agents])
        :param: is_target defines whether get q values from target network
        :return: q_vals is the value given by current value function. 
                (batch_size*num_agent*dim_actions if action is not given; batch_size*num_agent otherwise)
        """
        if isinstance(batch_obs, np.ndarray):   # Only used when choosing action
            batch_obs = torch.FloatTensor(batch_obs).to(self.device)
        if self.share_policy:
            if is_target:   # Target q networks
                q_vals = self.target_agent_q_nets[0](batch_obs) # batch_size*num_agents*dim_actions.
            else:   # Action q networks
                q_vals = self.agent_q_nets[0](batch_obs)    # batch_size*num*agents*dim_actions

            if action is not None:  # return q value of state-action pair
                action_batch = action.view(-1, 1)
                q_vals = torch.gather(q_vals, 1, action_batch)
        else:
            raise NotImplementedError("Not sharing policy for vdn method is not implemented, please change args.share_policy to true")
        return q_vals

    def append_transition(self, obs, action, reward, obs_):
        """Store transition"""
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate((self.obs2array(obs), action, [reward], self.obs2array(obs_)))
        self.buffer_ptr += 1

    def choose_action(self, obs, is_random=False):
        """
        Choose action according to obs.
        :param obs: observation (dict) with ["]
        :param is_random: (bool) whether randomly choose action
        """
        eps = np.random.uniform(0, 1)
        if is_random or eps >= self.e_greedy:
            return [np.random.choice(range(self.dim_action)) for _ in range(self.num_agents)]
        else:
            if self.share_policy:
                data = self.obs2data(obs)   # Convert obs(dict) to agent's network's input data(torch_geometric.Data)
                q_vals = self.agent_q_nets[0](data) # 5(num_nodes)*5(num_actions) represents the state-action values
                actions = torch.max(q_vals, dim=1)[1].tolist()
            else:
                raise NotImplementedError("Not sharing policy for vdn method is not implemented, please change args.share_policy to true")
            return actions

    def obs2array(self, obs):
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

    def obs2data(self, obs):
        """This function convert obs(dict) to network's input data(torch_geometric.Data)
        @params: 
            obs: observation (dict) check self.obs2array() to find out its contents. 
        @return:
            data: distributed agents' obervations. In torch_geometric.Data form
        """
        if type(obs) == dict:
            x = torch.FloatTensor(np.vstack([obs["idle_drivers"], obs["upcoming_cars"], obs["demands"]]).T)
            edge_attr = torch.FloatTensor(np.vstack([np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])]), 
                                                     self.len_mat]).T)
            data = Data(x=x, edge_attr=edge_attr, edge_index=torch.LongTensor(self.edge_index))
        elif type(obs)==np.ndarray:
            if len(obs.shape)==1:   # obs contains a single observation
                x = torch.FLoatTensor(obs[:self.dim_node_obs*self.num_agents].reshape(self.dim_node_obs, self.num_agents).T)
                node_state_end_idx = self.dim_node_obs*self.num_agents
                edge_attr = torch.FloatTensor(obs[node_state_end_idx : ].\
                                                  reshape(self.dim_edge_obs, self.edge_index.shape[1]).T)
                data = Data(x=x, edge_attr=edge_attr, edge_index=torch.LongTensor(self.edge_index))
            else:   # obs is a batch of observation
                batch_size = obs.shape[0]
                x = torch.FloatTensor(np.concatenate(
                        obs[:, :self.dim_node_obs*self.num_agents].\
                        reshape(-1, self.dim_node_obs, self.num_agents).swapaxes(1,2)))
                node_state_end_idx = self.dim_node_obs*self.num_agents
                edge_attr = torch.FloatTensor(np.concatenate(
                                obs[:, node_state_end_idx : ].\
                                reshape(-1, self.dim_edge_obs, self.edge_index.shape[1]).swapaxes(1,2)))
                num_edges = self.edge_index.shape[1]
                data_edge_index = np.concatenate([[self.edge_index+i*self.num_agents]\
                                                   for i in range(batch_size)]).swapaxes(0,1).reshape(self.dim_edge_obs,-1)
                data = Data(x=x, edge_attr=edge_attr, edge_index=torch.LongTensor(data_edge_index))
        return data
        

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

    def save_network(self):
        import datetime
        for idx, agent in enumerate(self.agent_q_nets):
            file_name = "net_"+str(idx)+"_"+datetime.datetime.now().strftime('%m%d_%H%M')
            torch.save(agent.state_dict(), self.save_dir+file_name+".pkl")
            print("Q-network saved in ", self.save_dir)

    def restore(self):
        for idx, agent in enumerate(self.agent_q_nets):
            path = self.save_dir + "\\net_"+str(idx)+".pkl"
            agent.load_state_dict(torch.load(path))
            print("Succesfully loaded q-network")