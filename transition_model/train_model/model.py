import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class Net(MessagePassing):
    def __init__(self, num_node, config, device, flow):
        super(Net, self).__init__(aggr='max', flow=flow)
        self.num_node = num_node
        in_channels = config["in_channels"]
        edge_attr_channels = config["edge_channels"]
        out_channels = config["out_channels"]
        message_hidden_layer = config["message_hidden_layer"]
        update_hidden_layer = config["update_hidden_layer"]
        self.dim_message = config["dim_message"]
        if not isinstance(message_hidden_layer, int):   # If the message hidden layer is one, then it must be two layers
            message_hid_layer_1 = message_hidden_layer[0]
            message_hid_layer_2 = message_hidden_layer[1]
            update_hidden_layer_1 = update_hidden_layer[0]
            update_hidden_layer_2 = update_hidden_layer[1]
        
        self.device = device
        self.num_node = num_node    # Number of nodes is the same as out_channels
        if not isinstance(message_hidden_layer, int):
            self.mlp_message = nn.Sequential(nn.Linear(2 * in_channels + edge_attr_channels, message_hid_layer_1),
                                            nn.ELU(),
                                            nn.Linear(message_hid_layer_1, message_hid_layer_2), 
                                            nn.ELU(),
                                            nn.Linear(message_hid_layer_2, self.dim_message)).to(self.device)  # Output a node's 'score'
            self.mlp_update = nn.Sequential(nn.Linear(self.dim_message*self.num_node, update_hidden_layer_1),
                                            nn.ELU(),
                                            nn.Linear(update_hidden_layer_1, update_hidden_layer_2), 
                                            nn.ELU(),
                                            nn.Linear(update_hidden_layer_2, out_channels)).to(self.device)  # Output distribution from n_node's score
        else:
            self.mlp_message = nn.Sequential(nn.Linear(2 * in_channels, message_hidden_layer), 
                                            nn.ELU(), 
                                            nn.Linear(message_hidden_layer, self.dim_message)).to(self.device)  # Output a node's 'score'
            self.mlp_update = nn.Sequential(nn.Linear(self.dim_message*self.num_node, update_hidden_layer), 
                                            nn.ELU(), 
                                            nn.Linear(update_hidden_layer, out_channels)).to(self.device)

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)  

        return 100*F.softmax(out, dim=1)    # Multiply by 100 to increase the loss (think it can accelerate training.)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp_message(tmp)

    def aggregate(self, inputs, edge_index):
        num_inputs = torch.max(edge_index[0])+1  # number of inputs, equals n_node*BATCH_SIZE
        # score = (torch.ones([num_inputs, self.num_node, self.dim_message])*-10000).to(self.device)  # To make unreachable places extremly small, we initialize each 'score' as -10000
        score = torch.zeros([num_inputs, self.num_node, self.dim_message]).to(self.device)  # To make unreachable places extremly small, we initialize each 'score' as -10000
        # score = torch.zeros([num_inputs, self.num_node, self.dim_message]).to(self.device)  # To make unreachable places extremly small, we initialize each 'score' as -10000
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