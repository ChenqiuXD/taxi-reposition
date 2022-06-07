import copy
import torch.nn as nn
import torch.optim
from agents.vdn.agent_q_function import AgentQFunction
from rl_algo.utils.base_agent import BaseAgent
import numpy as np
from torch_geometric.data import Data
import os


class VDN(BaseAgent):
    """
    Trainer class for QMix with MLP policies.
    """
    def __init__(self, args, env_config):
        super().__init__(args, env_config)

        self.share_policy = self.args.share_policy 
        self.edge_index = env_config["edge_index"]
        self.len_mat = env_config["len_mat"]
        self.dim_node_obs = env_config["dim_node_obs"]    # idle_drivers, upcoming_cars, demands
        self.dim_edge_obs = env_config["dim_edge_obs"]

        # Initialize buffer
        self.buffer_size = self.args.buffer_size
        self.buffer_ptr = 0
        self.buffer = np.zeros((self.buffer_size, 2*self.dim_obs+self.num_agents+2))
        # a transition is [s,a,r,s'], action is a [num_agent*1] vector, reward and time_step are both scalars

        # Initialize agent networks
        agent_q_config = {
            "num_node": self.num_agents,
            "dim_action": env_config['dim_action'],
            "dim_node_obs": self.dim_node_obs+1,    # node obs(3) plus time_step
            "dim_edge_obs": self.dim_edge_obs,
            "dim_message": 8,
            "out_channels": self.dim_action,
            "message_hidden_layer": [256,128],
            "update_hidden_layer": [64, 32],
        }
        if self.share_policy:
            self.agent_q_nets = [AgentQFunction(self.args, agent_q_config, id=0).to(self.device)]
        else:
            raise NotImplementedError("Not sharing policy for vdn method is not implemented, please change args.share_policy to true")

        # Initialize by xavier_uniform_
        for agent in self.agent_q_nets:
            for net in agent.modules():
                if isinstance(net, nn.Linear):
                    nn.init.normal_(net.weight, mean=0, std=0.05)
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

        self.optimizer = torch.optim.SGD(params=self.parameters, lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.save_dir = os.path.abspath('.')+"\\run_this\\agents\\vdn\\"

    def learn(self):
        """Update the network parameters using Q-learning"""
        self.prep_train()
        if self.share_policy:
            sample_index = np.random.choice(self.buffer_size, self.batch_size, replace=False)
            batch_memory = self.buffer[sample_index, :]
            batch_time_step = batch_memory[:, -1].reshape(-1,1)
            batch_state = self.obs2data(batch_memory[:, :self.dim_obs], batch_time_step).to(self.device)
            batch_action = torch.LongTensor(
                batch_memory[:, self.dim_obs: self.dim_obs+self.num_agents]).to(self.device)
            batch_reward = torch.FloatTensor(
                batch_memory[:, self.dim_obs+self.num_agents: self.dim_obs+self.num_agents+1]).to(self.device)
            batch_next_state = self.obs2data(batch_memory[:, -self.dim_obs-1:-1], batch_time_step+1).to(self.device)

            # Current Q values
            curr_agent_q_vals = self.get_q_value(batch_state, action=batch_action, is_target=False).view(self.batch_size, -1)
            curr_mixed_q = torch.sum(curr_agent_q_vals, dim=1).view(-1, 1)

            # Next state Q values
            next_agent_q_vals_all = self.get_q_value(batch_next_state, is_target=True)
            next_agent_q_vals= torch.max(next_agent_q_vals_all, dim=1)[0].view(self.batch_size, self.num_agents)
            next_mixed_q = torch.sum(next_agent_q_vals, dim=1).view(-1, 1)

            # Judge whether it is the last step
            is_last_step = torch.LongTensor(batch_time_step+1==self.episode_length).to(self.device)
            next_mixed_q -= next_mixed_q*is_last_step

            # Compute Bellman loss
            target_mixed_q = (batch_reward + next_mixed_q).detach()
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
        # Get q values by q-network
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
        self.buffer[self.buffer_ptr] = np.concatenate((self.obs2array(obs), action, [reward], self.obs2array(obs_), [obs['time_step']]))
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

    def obs2data(self, obs, batch_time_step=-1):
        """This function convert obs(dict) to network's input data(torch_geometric.Data)
        @params: 
            obs: observation (dict) check self.obs2array() to find out its contents. 
        @return:
            data: distributed agents' obervations. In torch_geometric.Data form
        """
        if type(obs) == dict:
            x = torch.FloatTensor(np.vstack([obs["idle_drivers"], obs["upcoming_cars"], obs["demands"], [obs["time_step"]]*self.num_agents]).T)
            edge_attr = torch.FloatTensor(np.vstack([np.array([obs["edge_traffic"][self.edge_index[0,j], self.edge_index[1,j]] for j in range(self.edge_index.shape[1])]), 
                                                     self.len_mat]).T)
            data = Data(x=x, edge_attr=edge_attr, edge_index=torch.LongTensor(self.edge_index))
        elif type(obs)==np.ndarray:
            assert (batch_time_step!=-1).all(), ("Batch time step equals -1 in obs2data function")
            if len(obs.shape)==1:   # obs contains a single observation
                raise NotImplementedError("training batch data should not be 0, plesae change batch size to more than 1")
            else:   # obs is a batch of observation
                batch_size = obs.shape[0]
                x = torch.FloatTensor(np.concatenate(
                        np.concatenate((obs[:, :self.dim_node_obs*self.num_agents],
                                        np.hstack([[step_i]*self.num_agents for step_i in batch_time_step]).T),
                                        axis=1).reshape(-1, self.dim_node_obs+1, self.num_agents).swapaxes(1,2)))
                node_state_end_idx = self.dim_node_obs*self.num_agents
                edge_attr = torch.FloatTensor(np.concatenate(
                                obs[:, node_state_end_idx : ].\
                                reshape(-1, self.dim_edge_obs, self.edge_index.shape[1]).swapaxes(1,2)))
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