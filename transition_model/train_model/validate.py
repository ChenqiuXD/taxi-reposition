import sys
import os
sys.path.append(os.getcwd())   # Added so as to use codes from rl_algo package in the parent dir
sys.path.append(os.getcwd()+"\\transition_model")
sys.path.append(os.getcwd()+"\\rl_algo")

from model import Net
import numpy as np
import torch
import os
from read_data import transform_data
from utils import split_data, get_normalization
from torch_geometric.loader import DataLoader
from transition_model.config import get_config
from transition_model.driversGame import Game
import copy, time

# Constants
BATCH_SIZE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Currently using ", device)

# Generate graph:
n_node = 5  # Number of nodes
in_node_channels = 4  # Node features: init_cars, upcoming cars, demands, bonuses
in_edge_channels = 2  # Edge features: distance, traffic flow
out_node_channels = n_node  # Represent the distribution propotion of idle drivers

# ---------------------------------------------------------------------------
# Load network 
# ---------------------------------------------------------------------------
# Load dataset
path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/train_model/data_cat.npy"
data = np.load(path, allow_pickle=True)

normalization_config = get_normalization(data)  # Get max and min of each data
# data_list = transform_data(data, normalization_config, doNormalize=True)  # Normalize
data_list = transform_data(data, normalization_config, doNormalize=False)  # do notNormalize

# Define train_loader
_, validate_data_list = split_data(data_list, 10)
validate_loader = DataLoader(dataset=validate_data_list, batch_size=len(validate_data_list), shuffle=False)

# Init the model
config = {
    "in_channels": in_node_channels, 
    "edge_channels": in_edge_channels, 
    "out_channels": out_node_channels, 
    "dim_message": 8,
    "message_hidden_layer": [128, 64],
    "update_hidden_layer": [64,32],
    # "message_hidden_layer": 32,
    # "update_hidden_layer": 32,
}
model = Net(n_node, config, device, flow='source_to_target').to(device)
model_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/train_model/tran_model.pkl"
model.load_state_dict(torch.load(model_path))



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Validation on validation set data
model.eval()
for data_validate in validate_loader:
    data_validate = data_validate.to(device)
    result = model(data_validate).detach().cpu().numpy()
    print("Real data is\n", data_validate)
    print("Predicted data is:\n", np.around(result, decimals=1))

result = model(data_validate).detach().cpu().numpy()
print("Real data is\n", data_validate)
print("Predicted data is:\n", np.around(result, decimals=1))

label = data_validate.y.cpu().detach().numpy()
print("THe max deviation is: ", np.max(label-result))
print("The mean deviation is: ", np.mean(label-result))

print("Done validating using generated data")



# Validation use newly generated data, choose data_list[idx] as test data
# data_validate = data_list[20].to(device)
lr = 0.05
MAX_BONUS = 5
CONVERGE_CRITERION = lr / 10     # Criterion of convergence
SEQ_CONVERGE_CRITERION = 10      # The update term is smaller than criterion SEQ_CONVERGE_CRITERION times, then we consider the algo converge
MAX_EPOCH = 200
sim_steps = 600
input_args = ['--epoch', str(MAX_EPOCH), '--lr', str(lr), '--max_bonus', str(MAX_BONUS), '--converge_criterion', str(CONVERGE_CRITERION), '--sim_steps', str(sim_steps), '--display']   # No display
print("Args are: ", input_args)
edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                       [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]])    # COO form of connection matrix
N_TEST_DATA = 1    # Test 4 times
for data_idx in range(N_TEST_DATA): 
    # Randomly genearate data
    node_initial_cars = np.floor(np.random.uniform(0.2, 1, n_node) * 500).astype(int)     # Initial cars (at least 5 cars)
    node_demand = np.floor(np.random.uniform(0.01, 1, n_node) * 300).astype(int)   # Demands with maximum 200
    edge_traffic = np.floor(np.random.uniform(1000, 10000, (n_node, n_node))).astype(int)  # Edge traffic
    node_upcoming = np.floor(np.random.uniform(-1, 1, n_node) * 100).astype(int)   # Upcoming cars approximately 50
    node_bonuses = np.floor(np.random.uniform(0, MAX_BONUS, n_node)).astype(int)   # bonuses for reaching each nodes

    # directly assign values
    # node_features = data_validate.x.cpu().numpy().T
    # node_initial_cars = np.floor(node_features[0]).astype(int)     # Initial cars (at least 5 cars)
    # node_demand = np.floor(node_features[1]).astype(int)   # Demands with maximum 200
    # node_upcoming = np.floor(node_features[2]).astype(int)   # Upcoming cars approximately 50
    # node_bonuses = np.floor(node_features[3]).astype(int)   # bonuses for reaching each nodes
    # data_edge_index = data_validate.edge_index.cpu().numpy() # Added self-loop edge_index
    # edge_attr = data_validate.edge_attr.cpu().numpy()
    # edge_traffic = np.zeros([n_node, n_node])
    # for i in range(edge_attr.shape[0]):
    #     edge_traffic[data_edge_index[0,i], data_edge_index[1,i]] = edge_attr[i,1]
    # edge_traffic = (edge_traffic /500.0*10000).astype(int)

    print("Initial cars: ", node_initial_cars)
    print("demands: ", node_demand)
    print("upcoming: ", node_upcoming)
    print("bonuses: ", node_bonuses)

    # simulation settings
    setting = {
        "NODE_NUM": n_node,  
        "EDGES": edge_index, 
        "NODE_INITIAL_CAR": node_initial_cars,   # Initial idle drivers at each nodes
        "NODE_DEMAND": node_demand,    # Demands of nodes at next time
        "NODE_UPCOMING": node_upcoming,   # Upcoming idle drivers of each nodes
        "NODE_BONUS": node_bonuses,  # Bonuses assigned by platform agent
        "EDGE_TRAFFIC" : edge_traffic  # Traffic flow at each edges
    }

    # Parse args
    all_args = get_config(input_args)
    num_node = setting['NODE_NUM']

    # Assign initial state
    max_epoch = all_args.epoch
    game = Game(setting, all_args)

    iter_cnt = 0
    is_converged = [False] * SEQ_CONVERGE_CRITERION
    converged_cnt = 0
    # Start simulation
    while iter_cnt<max_epoch:
        # Drivers chooses actions
        actions = game.choose_actions().astype(int)
        # print("Action is: \n", actions)

        # Simulate to observe outcomes
        actions_without_diagonal = copy.deepcopy(actions)  # Delete diagonal actions since they cost zero time and need not simulation
        np.fill_diagonal(actions_without_diagonal, 0)

        if iter_cnt % 5 ==0:
            try:
                time_mat = game.simulate_game(actions_without_diagonal)
                # print("time_mat is: \n", time_mat)
            except:
                try:
                    time.sleep(1)
                    time_mat = game.simulate_game(actions_without_diagonal)
                except:
                    try:
                        time.sleep(1)
                        time_mat = game.simulate_game(actions_without_diagonal)
                    except:                
                        is_error_occured = True
                        print("unknown error occured, saved data")
                        break

        # Calculate the payoff
        payoff = game.observe_payoff(time_mat, actions)
        # print("Payoff is \n", payoff )

        # Drivers Update their policies
        update_term = game.update_policy(payoff, actions)
    
        # Check convergence
        if game.check_convergence(update_term):
            # Assign is_converged to True
            is_converged[converged_cnt] = True
            converged_cnt += 1

            # If the update term has been been smaller than criteion five times
            if converged_cnt == SEQ_CONVERGE_CRITERION:
                for node in game.nodes:
                    node.value_table_traj = node.value_table_traj[:iter_cnt+1, :]
                is_converged = True
                break
        else:
            print("At iteration ", iter_cnt, " the max update term is: ", np.max(update_term))
        
        iter_cnt += 1

    # Simulated result
    print("The simulated result is: \n", actions)

    # Model generated result
    dict_data = {'init_cars': node_initial_cars,
                 'demand': node_demand, 
                 'upcoming cars': node_upcoming, 
                 'bonuses': node_bonuses, 
                 'traffic': edge_traffic/10000.0*500.0,
                 'edges': edge_index, 
                 'actions': np.zeros([n_node, n_node])} # Do not need actions, thus assign to be 0
    data = transform_data([dict_data], normalization_config=0, doNormalize=False)
    output = model(data[0]).detach().cpu().numpy()/100.0
    print("The model prediction output (of cars) is: \n", np.floor((node_initial_cars*output.T).T))

print("Done")