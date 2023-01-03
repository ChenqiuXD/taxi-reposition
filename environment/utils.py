import numpy as np

def softmax(x):
    """ Return a single array of soft-max result.  """
    return np.exp(x) / np.sum(np.exp(x))

def get_adj_mat(edges):
    """Function convert COO-form edges to adajency matrix
    params: 
        edges: (np.ndarray) A COO form of connection edges. 
    return: 
        adj_mat (np.ndarray) adajency matrix without self-loop
    """
    num_nodes = np.max(edges) + 1    # Index of python starts from 0 thus should +1
    adj_mat = np.zeros([num_nodes, num_nodes])
    for cnt, idx in enumerate(edges[0]):
        adj_mat[idx, edges[1, cnt]] = 1
    adj_mat = np.maximum(adj_mat, np.eye(num_nodes))

    return adj_mat

def add_self_loop(edge_index):
    """Add self-loop to edge_index"""
    num_nodes = np.max(edge_index)+1
    return np.insert(edge_index, [edge_index.shape[1]], [np.arange(num_nodes), np.arange(num_nodes)], axis=1)

def vec2mat(edge_index, vec):
    """ This function transform the vector in COO form to corresponding matrix form """
    assert vec.shape[0]==edge_index.shape[1], "vec.shape != edge_index.shape in function vec2mat"
    num_nodes = np.max(edge_index) + 1  # python starts from 0, thus plus 1
    mat = np.zeros([num_nodes, num_nodes])
    for i in range(edge_index.shape[1]):
        mat[edge_index[0,i], edge_index[1,i]] = vec[i]
    return mat
