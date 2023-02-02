import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import MessagePassing


class ValueNet(nn.Module):
    def __init__(self, net_config: dict):
        super(ValueNet, self).__init__()
        self.n_node, out_channels, dim_node_features, dim_edge_features,\
            self.dim_message, self.dim_update, dim_message_layer, dim_update_layer,\
            dim_output_layer, self.device = net_config.values()

        # Move to cuda
        self.mlp_message = torch.nn.Sequential().to(self.device)
        self.mlp_update = torch.nn.Sequential().to(self.device)
        self.mlp_output = torch.nn.Sequential().to(self.device)

        # Construct message layer
        self.mlp_message.add_module("layer", torch.nn.Linear(dim_node_features+dim_edge_features, dim_message_layer[0]))
        self.mlp_message.add_module("relu", torch.nn.ReLU())
        for idx, dim in enumerate(dim_message_layer[1:]):
            self.mlp_message.add_module("layer_{}".format(idx), torch.nn.Linear(dim_message_layer[idx], dim))
            self.mlp_message.add_module("relu{}".format(idx), torch.nn.ReLU())
        self.mlp_message.add_module("layer_last", torch.nn.Linear(dim_message_layer[-1], self.dim_message))

        # Construct update layer
        self.mlp_update.add_module("layer", torch.nn.Linear(self.dim_message*self.n_node, dim_update_layer[0]))
        self.mlp_update.add_module("relu", torch.nn.ReLU())
        for idx, dim in enumerate(dim_update_layer[1:]):
            self.mlp_update.add_module("layer_{}".format(idx), torch.nn.Linear(dim_update_layer[idx], dim))
            self.mlp_update.add_module("relu{}".format(idx), torch.nn.ReLU())
        self.mlp_update.add_module("layer_last", torch.nn.Linear(dim_update_layer[-1], self.dim_update))
        
        # Construct update layer
        self.mlp_output.add_module("layer", torch.nn.Linear(self.dim_update*self.n_node, dim_output_layer[0]))
        self.mlp_output.add_module("relu", torch.nn.ReLU())
        for idx, dim in enumerate(dim_output_layer[1:]):
            self.mlp_output.add_module("layer_{}".format(idx), torch.nn.Linear(dim_output_layer[idx], dim))
            self.mlp_output.add_module("relu{}".format(idx), torch.nn.ReLU())
        self.mlp_output.add_module("layer_last", torch.nn.Linear(dim_output_layer[-1], out_channels))

        # Move to cuda
        self.mlp_message.to(self.device)
        self.mlp_update.to(self.device)
        self.mlp_output.to(self.device)

    def forward(self, node_features, edge_features, edge_index):
        # Calculate the messages
        node_message_input = node_features[edge_index[0]]
        message = self.mlp_message( torch.cat([node_message_input, edge_features], dim=1) )

        # Calculate the aggregate result
        batch_mul_nodes = (torch.max(edge_index)+1).cpu().detach().numpy()
        aggregated_message = torch.zeros([batch_mul_nodes, self.n_node, self.dim_message]).to(self.device)
        for idx in range(edge_index.shape[1]):
            aggregated_message[edge_index[1,idx], edge_index[0,idx]%self.n_node] = message[idx]
        
        # Calculate the update results
        aggregated_message = aggregated_message.reshape(-1, self.dim_message*self.n_node)
        gnn_output = self.mlp_update(aggregated_message)

        # Calculate final output by mlp_output
        return self.mlp_output( gnn_output.reshape(-1, self.n_node*self.dim_update) )