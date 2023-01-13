import numpy as np

from environment.env import Env
from environment.utils import softmax, get_adj_mat

def make_env(args):
    
    """This function return an env which has similar functionality as gyms. """
    edge_index = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 1, 2, 3, 4], 
                           [1, 4, 3, 0, 4, 2, 1, 4, 3, 0, 4, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]])
    num_nodes = np.max(edge_index) + 1
    adj_mat = get_adj_mat(edge_index)
    adj_mat_without_diagnal = adj_mat - np.eye(num_nodes)
    len_vec = np.zeros(edge_index.shape[1])
    for i in range(len_vec.shape[0]):
        if edge_index[0,i]==edge_index[1,i]:
            len_vec[i] = 0
        else:
            len_vec[i] = 100 if edge_index[0,i]==4 or edge_index[1,i]==4 else 141.4

    # Randomly generate some data
    episode_length = args.episode_length
    normalize = lambda arr: arr/np.sum(arr)
    node_initial_cars = normalize( np.random.uniform(0.2, 1, num_nodes) ) * 1000
    node_demand = np.vstack( [ normalize(np.random.uniform(0.2, 1, num_nodes))*1000 for _ in range(episode_length) ] )
    node_distribute = get_demands_distribution(node_demand, adj_mat)
    edge_traffic = np.floor(np.random.uniform(1000, 10000, (episode_length, num_nodes, num_nodes)) * adj_mat_without_diagnal).astype(int)  # Edge traffic
    # upcoming_cars = np.floor(np.random.uniform(-1, 1, [episode_length, num_nodes]) * 100).astype(int)   # Upcoming cars approximately 50
    # TODO: upcoming_cars are currently zero. 
    upcoming_cars = np.zeros([episode_length, num_nodes])

    env_config = {
        "num_nodes": 5,
        "edge_index": edge_index, 
        "dim_node_obs": 3,  # [idle drivers, upcoming cars, demands]
        "dim_edge_obs": 2,  # [traffic flow density, length]
        "episode_length": episode_length,   

        "initial_drivers": node_initial_cars,   # Initial idle drivers  [n_node * 1]
        "node_demand": node_demand,             # Demands for each nodes, [EPISODE_LEN * n_node]
        "upcoming_cars": upcoming_cars,         # Initial upcoming cars [n_node * 1]
        "demand_distribution": node_distribute,     # Probability of demands distribute drivers to other nodes
        "edge_traffic": edge_traffic,           # Traffic at each edges, [EPISODE_LEN * n_node * n_node]
        "len_vec": len_vec,
    }

    return Env(env_config=env_config, args=args)

def get_demands_distribution(node_demand, adj_mat):
    """This function distribute the demands of each nodes
    @params: 
        node_demand: (ndarray, [episode_length, n_node]) demands at each node at each iteration
        adj_mat: (ndarray, [n_node, n_node]) adjcency matrix with self-loop
    """
    episode_length = node_demand.shape[0]
    n_node = adj_mat.shape[0]
    distribution = np.random.uniform(1,3,[episode_length, n_node, n_node]) * adj_mat + \
                   10000*(np.vstack([adj_mat[np.newaxis, :, :]-1]*episode_length))    # unreachable entries' values are -10000
    dist_mat = np.vstack( [ np.vstack(softmax(distribution[i][j]) for j in range(n_node)) ] for i in range(episode_length) )
    return dist_mat


    