import numpy as np
from torch import triplet_margin_loss
import torch.nn as nn
import matplotlib.pyplot as plt

def add_self_loop(edge_index):
    """Add self-loop to edge_index"""
    num_node = np.max(edge_index)+1
    return np.insert(edge_index, [edge_index.shape[1]], [np.arange(num_node), np.arange(num_node)], axis=1)

def get_adj_mat(edges):
    """Function convert COO-form edges to adajency matrix

    params: 
        edges: (np.ndarray) A COO form of connection edges. 
    return: 
        adj_mat (np.ndarray) adajency matrix without self-loop
    """
    num_node = np.max(edges) + 1    # Index of python starts from 0 thus should +1
    adj_mat = np.zeros([num_node, num_node])
    for cnt, idx in enumerate(edges[0]):
        adj_mat[idx, edges[1, cnt]] = 1

    return adj_mat

def calc_loss(output, truth):
    # loss = nn.CrossEntropyLoss()
    loss = nn.MSELoss()
    lossc = loss(output, truth)
    return lossc

def softmax(x):
    if len(x.shape)==1:
        return np.exp(x) / np.sum(np.exp(x))
    elif len(x.shape)==2:
        return np.vstack([np.exp(x[i]) / np.sum(np.exp(x[i])) for i in range(x.shape[0])])

def plot_loss(loss_train, loss_validation):
    assert len(loss_train) == len(loss_validation), 'During ploting loss function, the length of training set do not equals to validation set. s'
    n_iter = len(loss_train)
    plt.plot(range(n_iter), loss_train, 'b')
    plt.legend('train loss')
    plt.plot(range(n_iter), loss_validation, 'r')
    plt.legend('validate loss')
    # plt.show()
    plt.savefig("transition_model_result")

def split_data(data_list, k):
    """This function randomly split data into training set and validation set (k-1 : 1)"""
    assert type(k) is int, 'The k variable in split_data function is not an int'
    n_data = len(data_list)
    n_train = int(np.floor(n_data / k * (k-1)))
    train_idx = np.random.choice(np.arange(n_data), size=n_train, replace=False)
    validate_idx = np.array(list(set(np.arange(n_data))-set(train_idx)))

    return [data_list[i] for i in train_idx], [data_list[i] for i in validate_idx]

def get_normalization(data):
    """This function normalize the data"""
    # init_cars: row 0, demand: row 1, upcoming cars: row 2, bonuses: row 3, traffic: row 4
    normalization_config = {'init_cars':[1e4,0], 'demand':[1e4,0], 'upcoming cars':[1e4,0], 'bonuses':[1e4,0], 'traffic':[1e4,0]}   # 4 attributes, 2 for max(1) and min(0)
    attributes = ['init_cars', 'demand', 'upcoming cars', 'bonuses', 'traffic']
    for data_i in data:
        for attr_i in attributes:
            if np.min(data_i[attr_i])<normalization_config[attr_i][0]:
                normalization_config[attr_i][0] = np.min(data_i[attr_i])
            if np.max(data_i[attr_i])>normalization_config[attr_i][1]:
                normalization_config[attr_i][1] = np.max(data_i[attr_i])
    return normalization_config