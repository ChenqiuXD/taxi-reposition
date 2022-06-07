import torch
import numpy as np
from torch_geometric.data import Data
import torch

from utils import add_self_loop

def transform_data(data, normalization_config, doNormalize=False):
    """This function transform the data from npy to torch.Data
    @ params:
        data: (ndarray, BATCH_SIZE* dict) numpy array of lists, represents the data
        normalization_config: (dict ['init_cars', 'demand', 'upcoming cars', 'bonuses', 'traffic']*2) 
                                the min(0) and max(1) for each attributes(4 in total)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(data, list) or isinstance(data, np.ndarray):
        sample_data = data[0]
    elif isinstance(data, dict):
        sample_data = data
    else:
        raise RuntimeError("Data type should be either list or dict in transform_data() function, please check. ")
    edge_index = add_self_loop(sample_data['edges'])
    
    # We first obtain edge_index and len_mat which does not change. 
    n_edges = len(edge_index[0]) # number of edges
    len_mat = np.zeros(n_edges)
    for i in range(n_edges):
        if edge_index[0,i]==edge_index[1,i]:
            len_mat[i] = 0
        elif edge_index[0,i] == 4 or edge_index[1,i] == 4:
            if doNormalize:
                len_mat[i] = (100 - 0)/(141.4-0)    # Normalization
            else:
                len_mat[i] = 100
        else:
            if doNormalize:
                len_mat[i] = (141.4-0)/(141.4-0)    # Normalization
            else:
                len_mat[i] = 141.4
    
    # Then we transform data dependent features 
    data_list = []
    n_node = len(sample_data['init_cars'])  # Number of nodes
    for data_iter in data:
        if data_iter==0:
            continue
        if doNormalize: # Do normalization
            # Transform node_features and edge_features separately
            node_features = np.vstack([normalize(data_iter['init_cars'], normalization_config['init_cars']),
                                       normalize(data_iter['demand'], normalization_config["demand"]), 
                                       normalize(data_iter['upcoming cars'], normalization_config['upcoming cars']),
                                       normalize(data_iter['bonuses'], normalization_config['bonuses'])]).T
            traffic = np.zeros(n_edges)
            for i in range(n_edges):
                if edge_index[0,i]!=edge_index[1,i]:
                    traffic[i] = normalize(data_iter['traffic'][edge_index[0,i], edge_index[1,i]], normalization_config['traffic'])
            edge_features = np.vstack((len_mat, traffic)).T
        else:   # Do not do normalization
            # Transform node_features and edge_features separately
            node_features = np.vstack([data_iter['init_cars'],
                                       data_iter['demand'], 
                                       data_iter['upcoming cars'],
                                       data_iter['bonuses']]).T
            traffic = np.zeros(n_edges)
            for i in range(n_edges):
                if edge_index[0,i]!=edge_index[1,i]:
                    traffic[i] = data_iter['traffic'][edge_index[0,i], edge_index[1,i]]
            edge_features = np.vstack((len_mat, traffic)).T

        data_torch = Data(x=torch.FloatTensor(node_features).to(device), 
                        edge_attr=torch.FloatTensor(edge_features).to(device), 
                        edge_index=torch.LongTensor(edge_index).to(device))

        label = data_iter['actions']

        # For 3 kinds of labels: 1 100% label; 2 0. label; 3 true re-positioning cars label
        data_torch.y = torch.FloatTensor(100*np.vstack([label[i]/data_iter['init_cars'][i] for i in range(n_node)])).to(device)
        # data_torch.y = torch.FloatTensor(np.vstack([label[i]/data_iter['init_cars'][i] for i in range(n_node)])).to(device)
        # data_torch.y = torch.FloatTensor(label)

        # Append data to data_list
        data_list.append(data_torch)
        
    return data_list

def normalize(attr, normalization_config):
    """Linear mapping for attributes
    @ params:
        attr: (scalar or ndarray) the attribute(s)
        normalization_config: (ndarray) [0] is min and [1] is max
    """
    return (attr-normalization_config[0])/(normalization_config[1]-normalization_config[0])