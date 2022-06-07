import sys
import os
sys.path.append(os.getcwd())   # Added so as to use codes from rl_algo package in the parent dir
sys.path.append(os.getcwd()+"\\transition_model")
sys.path.append(os.getcwd()+"\\rl_algo")

from transition_model.node import Node
from transition_model.utils import get_adj_mat
import numpy as np
import os
from transition_model.config import get_config
from transition_model.sumo_util import SumoEnv
import matplotlib.pyplot as plt
import re   # Used to parse string
import copy
import datetime


class Game:
    """The game procedure class"""
    def __init__(self, setting, args):
        """Assign init 
        params:
            setting: sumo related parameters
            args: hyper-parameters 
        """
        # SUMO related args
        self.setting = setting
        self.init_settings(setting) # Assign simulation related parameters
        self.adj_mat = get_adj_mat(self.edges)    # Convert COO form connection to adjency matrix
        self.edge_len = self.adj_mat * 100.   # Each len is 100 meters long
        self.colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm']

        # add simulation related PATH variables
        # if 'SUMO_HOME' in os.environ:
        #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #     sys.path.append(tools)
        # else:
        #     sys.exit("please declare environment variable 'SUMO_HOME'")

        # hyper-parameters
        self.lr = args.lr
        self.max_epoch = args.epoch # equals num_env_steps
        self.is_display = args.display
        self.max_bonus = args.max_bonus # equals 5
        self.converge_criterion = args.converge_criterion   

        # Initialize env
        self.sumo = SumoEnv(setting, sim_steps=int(args.sim_steps))
        self.nodes = [Node(i, setting, self.lr, self.max_epoch, self.node_init_car[i], self.node_demand[i], self.node_upcoming_car[i],
                           self.node_bonus[i]) for i in range(self.num_node)]

    def init_settings(self, setting):
        """Change the settings of demand, initial cars, and other values"""
        self.num_node, self.edges, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args
    
    def update_node_states(self):
        """This function is used for env_runner. We change the node_init_car, node_demand and other attribute of Game.
         Then this funciton further assign those attributes to Nodes. """
        for cnt, node in enumerate(self.nodes):
            node.idle_drivers = self.node_init_car[cnt]
            node.demand = self.node_demand[cnt]
            node.upcoming_cars = self.node_upcoming_car[cnt]
            node.bonus = self.node_bonus[cnt]

    def choose_actions(self):
        """Drivers choose actions according to their policies"""
        actions = np.zeros([self.num_node, self.num_node])
        for i in range(actions.shape[0]):
            actions[i] = self.nodes[i].choose_action()
        return actions

    def simulate_game(self, actions):
        """Simulate and record transversing time
        params: None
        returns: a matrix [num_node, num_node] representing the simulated travelling time with [·]_{ij} if from node i to node j
        """
        time_mat = self.sumo.simulate(actions, self.is_display)
        return np.around(time_mat, decimals=3)

    def observe_payoff(self, time_mat, actions):
        """Used to calculate the overall payoff considering demands, travelling time and other factors
        params: time_mat: [num_node, num_node](np.ndarray) travelling time between nodes
        return: payoff: [num_node, num_node](np.ndarray) overall payoff with [·]_{ij} 
                        represents payoff of driving from node i to node j. 
                        It is calculated by c_idle_{j} + t_{ij} - r_j where 
                        c_idle_{j} : probability of being idel at node j. (multiply max_bonus for normalization)
                        t_{ij} : travelling time from node i to j. (·/max_time*max_bonus for normalization)
                        r_j : bonuses obtained, range from [0, max_bonus] (no normalization)
        """
        # Calculate idle
        node_all_cars = np.sum(actions, axis=0) + self.node_upcoming_car
        idle_prob = np.zeros(self.num_node)
        for i in range(self.num_node):
            if self.node_demand[i] < node_all_cars[i]:    # Sufficiently satisfied demand
                idle_prob[i] = (1 - self.node_demand[i] / node_all_cars[i]) if node_all_cars[i]>1e-3 else 1
            else:   # Idle cars are less than demand.
                idle_prob[i] = 0
        idle_cost = np.tile(idle_prob*self.max_bonus*0.7, (self.num_node, 1))   # For normalization, multiply by max_bonus

        # Calculate the travelling time
        max_time = np.max(time_mat)
        time_mat = time_mat / max_time * self.max_bonus*0.3 # For normalization, multipy by max_bonus

        # Calculate bonuses
        bonuses = np.tile(self.node_bonus, (self.num_node, 1))

        # Calculate the overall payoff (use adj_mat to eliminate the payoff of unrealizable edges)
        for i in range(self.adj_mat.shape[0]):  # add self_loop (cannot directly add np.eye(self.num_node) because sometimes, self.adj_mat has already added self-lopo)
            if self.adj_mat[i,i]==0:
                self.adj_mat[i,i]=1
        return (-idle_cost - time_mat + bonuses) * self.adj_mat
        # return (-idle_cost - time_mat + bonuses) * (self.adj_mat+np.eye(self.num_node))

    def update_policy(self, payoff, actions):
        """Drivers update their policies
        parrams: 
            payoff: [num_node, num_node] (np.ndarray) simulated payoff matrix with [·]_{ij} represents node i to node j
        """
        diff = np.zeros(self.num_node)
        for i in range(self.num_node):
            diff[i] = self.nodes[i].update_policy(payoff[i], actions[i])
        return diff

    def check_convergence(self, update_term):
        """Check whether drivers' policies converged"""
        if np.max(update_term) <= self.converge_criterion:
            return True
        else:
            return False
    
    def plot(self):
        """Plot the trajectory of each nodes"""
        plt.figure(figsize=[25,7.5])
        # plt.figure()
        x = range(self.nodes[0].value_table_traj.shape[0])
        for i in range(self.num_node):
            plt.subplot(2,3,i+1)
            for j in range(self.num_node):
                if j in self.nodes[i].neighbour_list:
                    plt.plot(x, self.nodes[i].value_table_traj[:, j], label='to node %d'%j, color=self.colors[j])
            plt.legend()
        
        # Plot the frame of original data
        plt.subplot(2,3,6)
        cols = ["node {}".format(str(i)) for i in range(self.num_node)]
        rows = ["init cars", "demands", "upcoming cars", "bonus", "traffic"]
        # Obtain data
        data = [[0]*self.num_node] * len(rows)
        data[0] = list(self.node_init_car)
        data[1] = list(self.node_demand)
        data[2] = list(self.node_upcoming_car)
        data[3] = list(self.node_bonus)
        traffic = list(re.sub(' +', ' ', str(self.edge_traffic[i]))[1:-1] for i in range(self.num_node))
        traffic = [re.sub(' ', '\n', traffic[i]).lstrip().rstrip() for i in range(self.num_node)]
        data[4] = traffic
        tab = plt.table(cellText=data, colLabels=cols, rowLabels=rows, loc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)    

        # Assign the height of last rows
        cell_dict = tab.get_celld()
        for i in range(len(cols)):
            cell_dict[(5,i)]._height = 0.4
            for j in range(len(rows)+1):
                cell_dict[(j, i)]._width = 0.15

        plt.axis('tight')
        # tab.scale(1,5)
        plt.axis('off')

    def get_data(self, data_list, iter_cnt, actions):
        """Return the data: initial_state -> drivers' final policies"""
        data = {"num_node": self.num_node, "edges": self.edges, "init_cars": self.node_init_car, "demand": self.node_demand,
                "upcoming cars": self.node_upcoming_car, "bonuses": self.node_bonus, "traffic": self.edge_traffic, "actions": actions}
        data_list[iter_cnt] = data

def main(setting, args):
    # Parse args
    all_args = get_config(args)
    num_node = setting['NODE_NUM']

    # Assign initial state
    max_epoch = all_args.epoch
    game = Game(setting, all_args)
    
    iter_cnt = 0
    SEQ_CONVERGE_CRITERION = 10
    is_converged = [False] * SEQ_CONVERGE_CRITERION
    converged_cnt = 0
    while iter_cnt<max_epoch:
        # Drivers chooses actions
        actions = game.choose_actions().astype(int)
        # print("Action is: \n", actions)

        if iter_cnt==68:
            print("Reached 68")

        # Simulate to observe outcomes
        actions_without_diagonal = copy.deepcopy(actions)  # Delete diagonal actions since they cost zero time and need not simulation
        np.fill_diagonal(actions_without_diagonal, 0)

        # Simulation
        time_mat = game.simulate_game(actions_without_diagonal)

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
    plt.suptitle(time_str)

    # Save pic
    # path = img_path + '\\' + str(data_idx) + '.png'
    print("Result converged")
    plt.savefig("debug_result.png")


if __name__ == "__main__":
    # hyper parameters 
    input_args = ['--epoch', '100', '--lr', '0.05', '--max_bonus', '5', '--converge_criterion', '0.005', '--display']
    print("Args are: ", input_args)
    
    # Settings
    setting = {
        "NODE_NUM": 5,  
        "EDGES": np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                           [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]]),    # COO form of connection matrix
        "NODE_INITIAL_CAR": [253, 313, 134, 233, 457],   # Initial idle drivers at each nodes
        "NODE_DEMAND": [244, 243, 195, 94, 262],    # Demands of nodes at next time
        "NODE_UPCOMING": [-71, 26, 81, 34, -56],   # Upcoming idle drivers of each nodes
        "NODE_BONUS": [1, 1, 0, 4, 4],  # Bonuses assigned by platform agent
        "EDGE_TRAFFIC" : np.array([[4456, 6309, 3909, 8345, 1007],
                                    [5343, 5963, 2183, 7530, 6214],
                                    [2187, 8585, 6484, 6515, 4600],
                                    [3892, 5023, 5627, 8985, 5995],
                                    [4659, 9845, 7178, 5865, 2344]])     # Traffic flow at each edges
    }

    if sys.argv[1:]:    # if code is run with --args (e.g. python driversGame.py --epoch 100 --lr 5e-4)
        main(setting, sys.argv[1:])
    else:   # if code is run without --args (e.g. python driversGame.py)
        main(setting, input_args)



# actions = np.array([[ 95,  26,   0,  42,  94],
#                     [ 36, 133,  13,   0, 135],
#                     [  0,  18,  25,  29,  66],
#                     [ 19,   0,   8, 129,  81],
#                     [ 38,  38,  18,  77, 291]])   (iter_cnt:68)

# array([[ 94,  26,   0,  42,  95],
#        [ 36, 132,  13,   0, 136],
#        [  0,  18,  26,  29,  65],
#        [ 19,   0,   8, 130,  80],
#        [ 38,  38,  18,  78, 290]])    (iter_cnt:69)

# array([[ 94,  26,   0,  43,  94],
#        [ 36, 133,  13,   0, 135],
#        [  0,  18,  26,  29,  65],
#        [ 19,   0,   8, 131,  79],
#        [ 38,  38,  18,  77, 291]])    (iter_cnt:70)

# array([[ 94,  26,   0,  43,  94],
#        [ 36, 131,  13,   0, 137],
#        [  0,  18,  26,  29,  65],
#        [ 19,   0,   8, 131,  79],
#        [ 38,  38,  17,  78, 291]])    (iter_cnt:71)