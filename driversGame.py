from utils import Node
from utils import get_adj_mat
import numpy as np
import sys
from config import get_config
from sumo_util import SumoEnv


class Game:
    """The game procedure class"""
    def __init__(self, setting, args):
        """Assign init values
        params:
            setting: sumo related parameters
            args: hyper-parameters 
        """
        # SUMO related args
        self.num_node, self.edges, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args
        adj_mat = get_adj_mat(self.edges)    # Convert COO form connection to adjency matrix
        self.edge_len = adj_mat * 100.   # Each len is 100 meters long

        # hyper-parameters
        self.lr = args.lr
        self.max_epoch = args.epoch
        self.is_display = args.display

        # Initialize env
        self.sumo = SumoEnv(setting)
        self.nodes = [Node(i, self.node_init_car[i], self.node_demand[i], self.node_upcoming_car[i],
                           self.node_bonus[i]) for i in range(self.num_node)]

    def set_init_values(self, setting):
        """Set initial values by settings
        params:
            setting: sumo related parameters
        """
        self.num_node, self.edges, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args
        adj_mat = get_adj_mat(self.edges)    # Convert COO form connection to adjency matrix
        self.edge_len = adj_mat * 100.   # Each len is 100 meters long

    def choose_actions(self):
        """Drivers choose actions according to their policies"""
        actions = np.zeros(self.num_node, self.num_node)
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

    def observe_payoff(self, time_mat):
        """Used to calculate the overall payoff considering demands, travelling time and other factors
        params: time_mat: [num_node, num_node](np.ndarray) travelling time between nodes
        return: payoff: [num_node, num_node](np.ndarray) overall payoff with [·]_{ij} 
                        represents payoff of driving from node i to node j
        """
        pass

    def update_policy(self, payoff):
        """Drivers update their policies
        parrams: 
            payoff: [num_node, num_node] (np.ndarray) simulated payoff matrix with [·]_{ij} represents node i to node j
        """
        pass

    def check_convergence():
        """Check whether drivers' policies converged"""
        pass

    def get_data(self):
        """Return the data: initial_state -> drivers' final policies"""
        pass

def main(setting, args):
    # Parse args
    all_args = get_config(args)

    # Assign initial state
    max_epoch = all_args.epoch
    game = Game(setting, all_args)
    
    iter_cnt = 0
    while iter_cnt<max_epoch:
        # Drivers chooses actions
        actions = game.choose_actions()

        # Simulate to observe outcomes
        time_mat = game.simulate_game(actions)
        payoff = game.observe_payoff(time_mat)

        # Drivers Update their policies
        game.update_policy(payoff)

        # Check convergence
        if game.check_convergence():
            print("Game converged")
            break
    
    # Return the final value
    game.get_data()


if __name__ == "__main__":
    # hyper parameters 
    input_args = ['--epoch', '10', '--lr', '1e-4', '--display']
    print("Args are: ", input_args)
    
    # Settings
    setting = {
        "NODE_NUM": 5,  
        "EDGES": np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                           [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]]),    # COO form of connection matrix
        "NODE_INITIAL_CAR": [100, 200, 2000, 10, 50],   # Initial idle drivers at each nodes
        "NODE_DEMAND": [50, 1000, 300, 100, 20],    # Demands of nodes at next time
        "NODE_UPCOMING": [0, 0, 0, 0, 0],   # Upcoming idle drivers of each nodes
        "NODE_BONUS": [0, 0, 0, 0, 0],  # Bonuses assigned by platform agent
        "EDGE_TRAFFIC" : np.random.uniform(0,5,(5,5))     # Traffic flow at each edges
    }

    if sys.argv[1:]:    # if code is run with --args (e.g. python driversGame.py --epoch 100 --lr 5e-4)
        main(setting, sys.argv[1:])
    else:   # if code is run without --args (e.g. python driversGame.py)
        main(setting, input_args)

