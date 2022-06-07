import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

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

def add_self_loop(edge_index):
    """Add self-loop to edge_index"""
    num_node = np.max(edge_index)+1
    return np.insert(edge_index, [edge_index.shape[1]], [np.arange(num_node), np.arange(num_node)], axis=1)
