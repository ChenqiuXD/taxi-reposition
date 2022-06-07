import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch_geometric.data as Data

class AgentQFunction(MessagePassing):
    """Single nodes' agent functions"""
    def __init__(self, args, q_config, id):
        """ initilization function for agent q function
        @params: 
            args: program input args (e.g. lr, episode_length ...)
            q_config: q-network related params (e.g. layer_1 width ...)"""
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_node = q_config["num_node"]    # 5, the number of nodes. 
        self.dim_action = q_config["dim_action"]    # 5, the number of possible bonus, [0,1,2,3,4]
        self.dim_node_obs = q_config["dim_node_obs"]    # 3, [idle drivers, upcoming cars, demands]
        self.dim_edge_obs = q_config["dim_edge_obs"]    # 2, [traffic flow, length]
        self.out_channel = q_config["out_channels"]  # 1, action, which is the bonus

        # Initialize the mlp
        self.dim_message = q_config["dim_message"]  # Output dimension of message mlp (=1)
        self.message_hidden_layer = q_config["message_hidden_layer"]    # num of encoder in message mlp
        self.update_hidden_layer = q_config["message_hidden_layer"] # num of encoder in update mlp
        if not isinstance(self.message_hidden_layer, int):
            self.mlp_message = nn.Sequential(nn.Linear(2*self.dim_node_obs+self.dim_edge_obs, self.message_hidden_layer[0]),
                                             nn.ReLU(),
                                             nn.Linear(self.message_hidden_layer[0], self.message_hidden_layer[1]),
                                             nn.ReLU(), 
                                             nn.Linear(self.message_hidden_layer[1], self.dim_message)).to(self.device)
            self.mlp_update = nn.Sequential(nn.Linear(self.dim_message*self.num_node, self.update_hidden_layer[0]),
                                            nn.ReLU(),
                                            nn.Linear(self.update_hidden_layer[0], self.update_hidden_layer[1]),
                                            nn.ReLU(),
                                            nn.Linear(self.update_hidden_layer[1], self.out_channel)).to(self.device)
        else:
            self.mlp_message = nn.Sequential(nn.Linear(2*self.dim_node_obs+self.dim_edge_obs, self.message_hidden_layer), 
                                            nn.ReLU(), 
                                            nn.Linear(self.message_hidden_layer, self.dim_message)).to(self.device)
            self.mlp_update = nn.Sequential(nn.Linear(self.dim_message*self.num_node, self.update_hidden_layer), 
                                            nn.ReLU(), 
                                            nn.Linear(self.update_hidden_layer, self.out_channel)).to(self.device)

    def forward(self, data):
        # assert()    # Check that input should be torch_geometric.Data.data type
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        return out  # Return individual agent's q-approximation

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp_message(tmp)

    def aggregate(self, inputs, edge_index):
        num_inputs = torch.max(edge_index[0])+1  # number of inputs, equals n_node*BATCH_SIZE
        # score = (torch.ones([num_inputs, self.num_node, self.dim_message])*-10000).to(self.device)  # To make unreachable places extremly small, we initialize each 'score' as -10000
        score = torch.zeros([num_inputs, self.num_node, self.dim_message]).to(self.device)
        for idx in range(num_inputs):
            neighbour_list = torch.where(edge_index[0]==idx)[0]
            for j_idx in neighbour_list:
                j_node = edge_index[1, j_idx] % self.num_node
                score[idx, j_node] = inputs[j_idx]
        return score

    def update(self, inputs):
        inputs = inputs.reshape(-1, self.dim_message*self.num_node)
        return self.mlp_update(inputs)
        # return inputs
    