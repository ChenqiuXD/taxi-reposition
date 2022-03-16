from driversGame import Game
import numpy as np
from config import get_config

# hyper parameters 
input_args = ['--epoch', '2000', '--lr', '0.01', '--max_bonus', '5', '--converge_criterion', '0.01', '--display']   # No display
# input_args = ['--epoch', '2000', '--lr', '0.01', '--max_bonus', '5', '--converge_criterion', '0.01']   # Display
print("Args are: ", input_args)

# Settings
setting = {
    "NODE_NUM": 5,  
    "EDGES": np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                        [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]]),    # COO form of connection matrix
    "NODE_INITIAL_CAR": [50, 100, 250, 40, 110],   # Initial idle drivers at each nodes
    "NODE_DEMAND": [0, 0, 0, 0, 0],    # Demands of nodes at next time
    "NODE_UPCOMING": [10, 20, 30, 40, 50],   # Upcoming idle drivers of each nodes
    "NODE_BONUS": [0, 3, 2, 4, 0],  # Bonuses assigned by platform agent
    "EDGE_TRAFFIC" : np.random.uniform(0,1000,(5,5))     # Traffic flow at each edges
}

# Parse args
all_args = get_config(input_args)
num_node = setting['NODE_NUM']

# Assign initial state
max_epoch = all_args.epoch
game = Game(setting, all_args)

iter_cnt = 0
while iter_cnt<max_epoch:
    # Drivers chooses actions
    actions = game.choose_actions().astype(int)

    # Simulate to observe outcomes
    actions_without_diagonal = actions  # Delete diagonal actions since they cost zero time and need not simulation
    np.fill_diagonal(actions_without_diagonal, 0)
    time_mat = game.simulate_game(actions_without_diagonal)
    payoff = game.observe_payoff(time_mat, actions)

    # Drivers Update their policies
    update_term = game.update_policy(payoff)

    # Check convergence
    if game.check_convergence(update_term):
        for i in range(game.num_node):
            game.nodes[i].value_table = game.nodes[i].value_table[:iter_cnt, :]
        break
    else:
        print("At iteration ", iter_cnt, " the max update term is: ", np.max(update_term))
    
    iter_cnt += 1

# Return the final value
game.plot()
# game.get_data()