import numpy as np
from transition_model.driversGame import Game
from transition_model.config import get_config
from transition_model.utils import get_adj_mat
import matplotlib.pyplot as plt
import copy
import os
from pathlib import Path
import datetime

# Load data     THE DATA is copied from run13 data_seed_1.npy
data = np.load("data.npy", allow_pickle=True)
data = data[1]

# hyper parameters 
# Used SEED 10
SEED = 1
MAX_EPOCH = 200
sim_steps = 600
SEQ_CONVERGE_CRITERION = 10
 
lr = 0.05
MAX_BONUS = 5
CONVERGE_CRITERION = lr / 10     # Criterion of convergence
input_args = ['--epoch', str(MAX_EPOCH), '--lr', str(lr), '--max_bonus', str(MAX_BONUS), '--converge_criterion', str(CONVERGE_CRITERION), '--sim_steps', str(sim_steps), '--display']   # No display
# input_args = ['--epoch', str(MAX_EPOCH), '--lr', str(lr), '--max_bonus', str(MAX_BONUS), '--converge_criterion', str(CONVERGE_CRITERION), '--sim_steps', str(sim_steps)]    # Display
print("Args are: ", input_args)

# Fixed settings
num_node = 5
edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                       [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]])    # COO form of connection matrix
adj_mat = get_adj_mat(edge_index)

np.random.seed(SEED)
# Start simulation
# Settings
node_initial_cars = data["init_cars"]
node_demand = data['demand']
edge_traffic = data['traffic']
node_upcoming = data['upcoming cars']
node_bonuses = data['bonuses']

setting = {
    "NODE_NUM": num_node,  
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

for iter in range(5):
    # Assign initial state
    max_epoch = all_args.epoch
    game = Game(setting, all_args)

    iter_cnt = 0
    is_converged = [False] * SEQ_CONVERGE_CRITERION
    converged_cnt = 0
    while iter_cnt<max_epoch:
        # Drivers chooses actions
        actions = game.choose_actions().astype(int)
        # print("Action is: \n", actions)

        # Simulate to observe outcomes
        actions_without_diagonal = copy.deepcopy(actions)  # Delete diagonal actions since they cost zero time and need not simulation
        np.fill_diagonal(actions_without_diagonal, 0)

        time_mat = game.simulate_game(actions_without_diagonal)
        # print("time_mat is: \n", time_mat)

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

    # Return the final value
    # game.plot()
    # ts = datetime.datetime.now()
    # time_str = ts.strftime('%m %d - %H:%M')
    # plt.suptitle(ts)
    # plt.savefig('result'+str(iter)+'.png')
    print(actions)
