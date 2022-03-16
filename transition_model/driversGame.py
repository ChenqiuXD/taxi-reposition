from time import time
from Node import Node
from utils import get_adj_mat
import numpy as np
import sys
import os
from config import get_config
from sumo_util import SumoEnv
import matplotlib.pyplot as plt


class Game:
    """The game procedure class"""
    def __init__(self, setting, args):
        """Assign init values
        params:
            setting: sumo related parameters
            args: hyper-parameters 
        """
        # SUMO related args
        self.init_settings(setting) # Assign simulation related parameters
        adj_mat = get_adj_mat(self.edges)    # Convert COO form connection to adjency matrix
        self.edge_len = adj_mat * 100.   # Each len is 100 meters long

        # add simulation related PATH variables
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # hyper-parameters
        self.lr = args.lr
        self.max_epoch = args.epoch
        self.is_display = args.display
        self.max_bonus = args.max_bonus

        # Initialize env
        self.sumo = SumoEnv(setting)
        self.nodes = [Node(i, setting, self.lr, self.max_epoch, self.node_init_car[i], self.node_demand[i], self.node_upcoming_car[i],
                           self.node_bonus[i]) for i in range(self.num_node)]

    def init_settings(self, setting):
        """Change the settings of demand, initial cars, and other values"""
        self.num_node, self.edges, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args

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
        return time_mat

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
            if self.node_demand[i] <= node_all_cars[i]:    # Sufficiently satisfied demand
                idle_prob[i] = (1 - self.node_demand[i] / node_all_cars[i])
            else:   # Idle cars are less than demand.
                idle_prob[i] = 0
        idle_cost = np.tile(idle_prob*self.max_bonus, (self.num_node, 1))   # For normalization, multiply by max_bonus

        # Calculate the travelling time
        max_time = np.max(time_mat)
        time_mat = time_mat / max_time * self.max_bonus # For normalization, multipy by max_bonus

        # Calculate bonuses
        bonuses = np.tile(self.node_bonus, (self.num_node, 1))

        # Calculate the overall payoff
        return -idle_cost - time_mat + bonuses

    def update_policy(self, payoff):
        """Drivers update their policies
        parrams: 
            payoff: [num_node, num_node] (np.ndarray) simulated payoff matrix with [·]_{ij} represents node i to node j
        """
        diff = np.zeros(self.num_node)
        for i in range(self.num_node):
            diff[i] = self.nodes[i].update_policy(payoff[i])
        return diff

    def check_convergence(self, update_term):
        """Check whether drivers' policies converged"""
        if np.max(update_term) <= 1e-3:
            return True
        else:
            return False
    
    def plot(self):
        """Plot the trajectory of each nodes"""
        plt.figure()
        x = range(self.nodes[0].value_table.shape[0])
        for i in range(self.num_node):
            plt.subplot(2,3,i+1)
            for j in range(self.num_node):
                plt.plot(x, self.nodes[i].value_table[:, j])
        plt.show()

    def get_data(self):
        """Return the data: initial_state -> drivers' final policies"""
        print("Getting data")
        return -1

def main(setting, args):
    # Parse args
    all_args = get_config(args)
    num_node = setting['NODE_NUM']

    # Assign initial state
    max_epoch = all_args.epoch
    game = Game(setting, all_args)
    
    iter_cnt = 0
    while iter_cnt<max_epoch:
        is_converged = game.run(iter_cnt)
        if is_converged:
            print("Game converged")
            break
        
        iter_cnt += 1

    # Return the final value
    game.plot()
    return game.get_data()


if __name__ == "__main__":
    # hyper parameters 
    input_args = ['--epoch', '2000', '--lr', '0.01', '--max_bonus', '5', '--converge_criterion', '0.001', '--display']
    print("Args are: ", input_args)
    
    # Settings
    setting = {
        "NODE_NUM": 5,  
        "EDGES": np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                           [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]]),    # COO form of connection matrix
        "NODE_INITIAL_CAR": [100, 200, 2000, 10, 50],   # Initial idle drivers at each nodes
        "NODE_DEMAND": [50, 1000, 300, 100, 20],    # Demands of nodes at next time
        "NODE_UPCOMING": [10, 20, 30, 40, 50],   # Upcoming idle drivers of each nodes
        "NODE_BONUS": [0, 3, 2, 4, 0],  # Bonuses assigned by platform agent
        "EDGE_TRAFFIC" : np.random.uniform(0,5,(5,5))     # Traffic flow at each edges
    }

    if sys.argv[1:]:    # if code is run with --args (e.g. python driversGame.py --epoch 100 --lr 5e-4)
        main(setting, sys.argv[1:])
    else:   # if code is run without --args (e.g. python driversGame.py)
        main(setting, input_args)

