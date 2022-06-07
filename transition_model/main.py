import sys
import os
sys.path.append(os.getcwd())   # Added so as to use codes from rl_algo package in the parent dir
sys.path.append(os.getcwd()+"\\transition_model")
sys.path.append(os.getcwd()+"\\rl_algo")

from transition_model.driversGame import Game
import numpy as np
from transition_model.config import get_config
from transition_model.utils import get_adj_mat
import matplotlib.pyplot as plt
import copy
import os
from pathlib import Path
import datetime
import time

# hyper parameters 
# Used SEED 10
SEED = 1
MAX_EPOCH = 200
N_DATA = 200
sim_steps = 600
data = [0] * N_DATA

lr = 0.05
MAX_BONUS = 5
CONVERGE_CRITERION = lr / 10     # Criterion of convergence
SEQ_CONVERGE_CRITERION = 10      # The update term is smaller than criterion SEQ_CONVERGE_CRITERION times, then we consider the algo converge
input_args = ['--epoch', str(MAX_EPOCH), '--lr', str(lr), '--max_bonus', str(MAX_BONUS), '--converge_criterion', str(CONVERGE_CRITERION), '--sim_steps', str(sim_steps), '--display']   # No display
# input_args = ['--epoch', str(MAX_EPOCH), '--lr', str(lr), '--max_bonus', str(MAX_BONUS), '--converge_criterion', str(CONVERGE_CRITERION), '--sim_steps', str(sim_steps)]    # Display
print("Args are: ", input_args)

# Create run folder
run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/transition_model/runs")
if not run_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
    if len(exst_run_nums)==0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums)+1)
run_dir = run_dir / curr_run
if not run_dir.exists():
    os.makedirs(str(run_dir))
    print("Create runs folder", str(run_dir))
    img_path = str(run_dir) + '\\image_result'
    os.makedirs(img_path)

# Fixed settings
num_node = 5
edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                    [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]])    # COO form of connection matrix
adj_mat = get_adj_mat(edge_index)

np.random.seed(SEED)
is_error_occured = False
# Start generating datas
for data_idx in range(N_DATA):
    # Settings
    # node_initial_cars = np.floor(np.random.uniform(0.2, 1, num_node) * 500).astype(int)     # Initial cars (at least 5 cars)
    # node_demand = np.floor(np.random.uniform(0.01, 1, num_node) * 300).astype(int)   # Demands with maximum 200
    # edge_traffic = np.floor(np.random.uniform(1000, 10000, (num_node, num_node))).astype(int)  # Edge traffic
    # node_upcoming = np.floor(np.random.uniform(-1, 1, num_node) * 100).astype(int)   # Upcoming cars approximately 50
    # node_bonuses = np.floor(np.random.uniform(0, MAX_BONUS, num_node)).astype(int)   # bonuses for reaching each nodes

    # Initial try, fixed parameters
    node_initial_cars = np.array([408, 108, 353, 399, 299])
    node_demand = np.array([67, 59, 228, 50, 26])
    node_upcoming = np.array([19, 80, 6, 18, -93])
    node_bonuses = np.array([2, 2, 3, 0, 3])
    edge_traffic = np.array([[  0, 2000,   0, 1400, 6600],
                            [5900,   0, 2800,   0,  4160],
                            [  0,  3660,   0, 4000, 2480],
                            [4500,   0, 8400,   0, 4457],
                            [9500, 9888, 5100, 8435,   0]])

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

    # Return the final value
    game.plot()
    ts = datetime.datetime.now()
    time_str = ts.strftime('%m %d - %H:%M')
    plt.suptitle(ts)

    # Save pic
    path = img_path + '\\' + str(data_idx) + '.png'
    plt.savefig(path)
    plt.close()

    # Save data
    game.get_data(data, data_idx, actions)
    data[data_idx]['is_converged'] = is_converged

    # If error occured, break the for loop
    if is_error_occured:
        break

    # Save data every 10 iterations
    if data_idx % 10 == 9:
        path = str(run_dir) + '\\data_seed_'+ str(SEED)+'_idx_'+str(data_idx) + '.npy'
        np.save(path, data)

print("Done")



# prediction
# array([[[15.537453  , 26.110855  ,  0.96627736, 36.248596  ,   21.136812  ],
#         [ 6.111378  , 68.75438   ,  8.501771  ,  0.27047914,   16.361982  ],
#         [ 0.8879243 , 22.109299  , 26.707708  , 35.841583  ,   14.4534855 ],
#         [ 2.236189  ,  0.4664547 ,  7.46256   , 78.76358   ,   11.071216  ],
#         [ 2.8700902 , 21.376682  , 10.033957  , 45.344524  ,   20.374752  ]]], dtype=float32)


# Prediction
# array([[[41.737, 19.036,  0.523,  4.991, 33.714],
#         [13.09 , 24.544, 40.817,  0.904, 20.646],
#         [ 0.087,  8.172, 76.079,  2.849, 12.813],
#         [13.352,  0.507, 46.174, 11.14 , 28.828],
#         [ 9.711, 15.323, 40.274,  5.031, 29.661]]], dtype=float32)
# array([[50.        , 14.80582524,  0.        ,  4.12621359, 31.06796117],
#        [ 8.92857143, 35.71428571, 33.92857143,  0.        , 21.42857143],
#        [ 0.        ,  5.88235294, 80.95238095,  1.68067227, 11.48459384],
#        [10.66997519,  0.        , 48.38709677, 14.39205955, 26.55086849],
#        [ 7.56578947,  9.53947368, 40.13157895,  3.28947368, 39.47368421]])