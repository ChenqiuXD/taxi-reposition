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

        # hyper-parameters
        self.lr_drivers = args.lr_drivers
        self.max_epoch = args.num_env_steps
        # self.is_display = args.render
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        # Initialize env
        self.nodes = [Node(i, setting, self.lr_drivers, max_epoch=self.max_epoch, warmup_epoch=args.warmup_steps) for i in range(self.num_nodes)]

    def init_settings(self, setting):
        """Change the settings of demand, initial cars, and other values"""
        self.num_nodes, self.edges, self.len_mat, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()   # Unpack args
    
    def update_state(self, state):
        """Update state received from Env"""
        self.node_init_car = state["idle_drivers"]
        self.node_demand = state["demands"]
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
            actions[i] = np.floor(prob*self.node_init_car[i])
            
            # Assign remaining drivers
            remain_cars = self.node_init_car[i] - np.sum(actions[i])
            for _ in range(int(remain_cars)):
                idx = np.random.choice(np.arange(self.num_nodes), p=prob)
                actions[i, idx] += 1 

        return actions

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
        norm_factor = [0.15, 0.35, 1]
        # Calculate idle drivers
        node_all_cars = np.sum(actions, axis=0) + self.node_upcoming_car
        idle_cost = -np.tile(node_all_cars/self.node_demand* (self.max_bonus-self.min_bonus) *norm_factor[0],
                            (self.num_nodes, 1))   # For normalization, multiply by max_bonus

        # Calculate the travelling time
        max_time = np.max(time_mat)
        time_mat = -time_mat / max_time *  (self.max_bonus-self.min_bonus)  * norm_factor[1] # For normalization, multipy by max_bonus

        # Calculate bonuses
        bonuses = np.tile(bonuses, (self.num_nodes, 1)) * norm_factor[2]

        # Calculate the overall payoff (use adj_mat to eliminate the payoff of unrealizable edges)
        for i in range(self.adj_mat.shape[0]):  # add self_loop (cannot directly add np.eye(self.num_nodes) because sometimes, self.adj_mat has already added self-lopo)
            if self.adj_mat[i,i]==0:
                self.adj_mat[i,i]=1
        return (idle_cost + time_mat + bonuses) * self.adj_mat

    def update_policy(self, payoff):
        """Drivers update their policies
        parrams: 
            payoff: [num_nodes, num_nodes] (np.ndarray) simulated payoff matrix with [·]_{ij} represents node i to node j
        """
        diff = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            diff[i] = self.nodes[i].update_policy(payoff[i])
        return diff

    def check_convergence(self, update_term):
        """Check whether drivers' policies converged"""
        if np.max(update_term) <= self.converge_criterion:
            return True
        else:
            return False
    
    def get_nodes_actions(self):
        return np.array([self.nodes[i].choose_action() for i in range(self.num_nodes)])

    def get_state(self):
        return self.node_init_car