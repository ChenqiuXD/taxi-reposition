from environment.node import Node
from environment.utils import get_adj_mat
import numpy as np


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
        self.neighbour_list = [np.where(self.adj_mat[i]==1)[0] for i in range(self.num_nodes)]

        # hyper-parameters
        self.max_epoch = args.num_env_steps
        # self.is_display = args.render
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        # learning_rate config
        self.lr_drivers = args.lr_drivers
        self.min_lr_drivers = args.min_lr_drivers
        self.decrease_lr_drivers = args.decrease_lr_drivers

        # Initialize env
        self.nodes = [Node(i, setting, self.lr_drivers, max_epoch=self.max_epoch, warmup_epoch=args.warmup_steps) for i in range(self.num_nodes)]

    def init_settings(self, setting):
        """Change the settings of demand, initial cars, and other values"""
        self.num_nodes, self.edges, self.len_mat, self.node_init_car, self.node_demands, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args
    
    def update_state(self, state):
        """Update state received from Env"""
        self.node_init_car = state["idle_drivers"]
        self.node_demands = state["demands"]
        self.node_upcoming_car = state["upcoming_cars"]
        self.edge_traffic = state["edge_traffic"]
        for i in range(self.num_nodes):
            self.nodes[i].idle_drivers = state["idle_drivers"][i]

    def choose_actions(self):
        """Drivers choose actions according to their policies"""
        actions = np.zeros([self.num_nodes, self.num_nodes])
        for i in range(actions.shape[0]):
            prob = self.nodes[i].choose_action()  # Probability of driving to neighbour nodes

            # Calculate the number of drivers
            actions[i][self.neighbour_list[i]] = np.floor(prob*self.node_init_car[i])

            # Assign remaining drivers
            remain_cars = self.node_init_car[i] - np.sum(actions[i])
            for _ in range(int(remain_cars)):
                idx = np.random.choice(self.neighbour_list[i], p=prob)
                actions[i, idx] += 1

        actions = np.maximum(actions, 1)    # To make every element not zero. Since there are inverse of drivers' policies in DirectAgent.learn() function
        return actions*self.adj_mat     # Eliminate the unreachable nodes. P.S. Since unreachable elements would be one by line 51, here we ensure they are zero. 

    def observe_payoff(self, bonuses, time_mat, actions):
        """Used to calculate the overall payoff considering demands, travelling time and other factors
        params: time_mat: [num_nodes, num_nodes](np.ndarray) travelling time between nodes
        return: payoff: [num_nodes, num_nodes](np.ndarray) overall payoff with [·]_{ij} 
                        represents payoff of driving from node i to node j. 
                        It is calculated by c_idle_{j} + t_{ij} - r_j where 
                        c_idle_{j} : probability of being idel at node j. (multiply max_bonus for normalization)
                        t_{ij} : travelling time from node i to j. (·/max_time*max_bonus for normalization)
                        r_j : bonuses obtained, range from [0, max_bonus] (no normalization)
        """
        factor = {"idle": 1, "time": 0.5, "bonuses": 0.25, 'entropy': 0.2}
        # Calculate idle drivers
        # node_all_cars = np.sum(actions, axis=0) + self.node_upcoming_car
        node_all_cars = np.sum(actions, axis=0)
        idle_cost = np.vstack([node_all_cars/self.node_demands]*5)
        # idle_cost = np.tile(node_all_cars/self.node_demands,
        #                     (self.num_nodes, 1))   # For normalization, multiply by max_bonus

        # Calculate the travelling time
        time_cost = np.zeros([self.num_nodes, self.num_nodes])
        for i in range(self.num_nodes):
            time_cost[i] = ( time_mat[i]-np.min(time_mat[i]) ) / ( np.max(time_mat[i])-np.min(time_mat[i]) )

        # Calculate bonuses
        # bonuses = np.tile(bonuses, (self.num_nodes, 1))
        bonuses = np.vstack([bonuses]*self.num_nodes)

        # Calculate the entropy cost
        drivers_policies = np.array([ dist / np.sum(dist) for dist in actions ])
        entropy_cost = np.zeros([ self.num_nodes, self.num_nodes ])
        for i in range(self.num_nodes):
            entropy_cost[i][self.neighbour_list[i]] = np.log(drivers_policies[i][self.neighbour_list[i]])

        # Calculate the overall payoff (use adj_mat to eliminate the payoff of unrealizable edges)
        payoff = - factor["idle"] * idle_cost - factor['time'] * time_cost\
                 + factor["bonuses"] * bonuses - factor['entropy'] * entropy_cost
        return payoff

    def decrease_lr(self):
        """ Decrease the lower-level agents' learning rate """
        for node in self.nodes:
            node.lr = np.maximum( node.lr-self.decrease_lr_drivers, self.min_lr_drivers )

    def update_policy(self, bonuses, time_mat, nodes_actions):
        """Drivers update their policies
        parrams: 
            payoff: [num_nodes, num_nodes] (np.ndarray) simulated payoff matrix with [·]_{ij} represents node i to node j
        """
        payoff = self.observe_payoff(bonuses, time_mat, nodes_actions)
        for i in range(self.num_nodes):
            self.nodes[i].update_policy(payoff[i][self.neighbour_list[i]])
    
    def get_nodes_actions(self):
        actions = np.zeros([self.num_nodes, self.num_nodes])
        for i in range(self.num_nodes):
            actions[i][self.neighbour_list[i]] = self.nodes[i].choose_action()
        return actions

    def get_state(self):
        return self.node_init_car