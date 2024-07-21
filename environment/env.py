import numpy as np
from environment.utils import get_adj_mat, vec2mat
from environment.game_model import Game 

class Env:
    """ The reinforcement learning environment """
    def __init__(self, env_config, args):
        """ This function initialize the environment
        INPUT:          env_config: (dict) environment related informatino. e.g. num_nodes, length_matrix
         """
        self.env_config = env_config
        self.edge_index = self.env_config["edge_index"]
        self.adj_mat = get_adj_mat(self.edge_index)
        self.num_nodes = self.env_config["num_nodes"]    # number of nodes
        self.len_vec = self.env_config["len_vec"]
        self.len_mat = vec2mat(self.edge_index, self.len_vec)
        self.dim_node_obs = self.env_config["dim_node_obs"]
        self.dim_edge_obs = self.env_config["dim_edge_obs"]
        self.episode_length = args.episode_length

        self.dim_action = self.num_nodes
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus
        self.obs_spaces = self.dim_node_obs * self.num_nodes + self.dim_edge_obs * self.edge_index.shape[1]
        # total observation dimension =  node_obs * num_nodes + edge_obs * num_edges

        # flag of whether reset
        self.is_reset = False
        self.is_two_loop = args.is_two_loop
        self.output_path = args.output_path

        # Drivers' Game, used to simulate the payoff
        setting = {
            "num_nodes": self.num_nodes,  
            "edge_index": self.edge_index, 
            "len_mat": self.len_mat, 
            "node_initial_cars": self.env_config["initial_drivers"], # initial idle drivers  [num_nodes*1]
            "node_demands": self.env_config["node_demand"][0],       # initial demands at each node  [num_nodes*1]
            "node_upcoming": self.env_config["upcoming_cars"][0],   # upcoming cars [num_nodes * 1]
            "node_bonuses": [0 for _ in range(self.num_nodes)],          # Bonuses assigned by platform agent, currently not assigned
            "edge_traiffc" : self.env_config["edge_traffic"][0],    # edge_traffic at time 0 [num_nodes * num_nodes]
        }

        self.games = [Game(setting, args) for _ in range(self.episode_length)]    # For time_step in range(episode_length), there is a game
        self.cur_state = None # Initialize in self.reset()

    def reset(self):
        """ Reset the envirnonment. Usually used when start a new episode """
        self.is_reset = True

        # Get initial state
        init_state = {  # State format: idle_drivers, upcoming_cars, demands, edge_traffic, time_step
            "idle_drivers": self.env_config["initial_drivers"], # initial idle drivers  [num_nodes*1]
            "upcoming_cars": self.env_config["upcoming_cars"][0],  # upcoming cars [num_nodes * 1]
            "demands": self.env_config["node_demand"][0],       # initial demands at each node  [num_nodes*1]
            "edge_traffic": self.env_config["edge_traffic"][0],  # edge_traffic at time 0 [num_nodes * num_nodes]
            "len_mat": self.len_mat, 
            "time_step": 0
        }
        self.cur_state = init_state

        # Set initial state's variables
        self.games[0].update_state(init_state)

        return init_state

    def step(self, bonuses):
        """ Step to next state.
        INPUT:      actions: (num_nodes*1 np.ndarray) is the bonuses assigned to each node
        OUTPUT:     next_state: next state
                    reward: MDP defined reward
                    done: (0/1) means whether the episode has ended
         """
        if not self.is_reset:
            raise RuntimeError("env is not reset but step function is called.")
        
        # agents choose actions
        time_step = self.cur_state["time_step"]
        self.games[time_step].update_state(self.cur_state)

        # Update agents policy
        time_mat = self.get_time_mat(self.cur_state["len_mat"], self.cur_state["edge_traffic"], self.adj_mat)
        if self.is_two_loop:    # double loop, the inner loop computes the nash equilibrium. 
            for t in range(50):
                nodes_actions = self.games[time_step].choose_actions()
                self.games[time_step].update_policy(bonuses, time_mat, nodes_actions)
        else:
            nodes_actions = self.games[time_step].choose_actions()
            self.games[time_step].update_policy(bonuses, time_mat, nodes_actions)

        # NOTE that agents choose actions twice, so that the bonuses would immediately impact on the agents' policies. 
        # Choose actions according to the newly updated agents policies. 
        nodes_actions = self.games[time_step].choose_actions()

        # Prepare information of next state
        if self.cur_state["time_step"]+1 >= self.episode_length:
            self.is_reset = False
            done = True
        else:
            time_step = self.cur_state["time_step"] + 1
            done = False

        # Calculate reward for drivers agents: returns a list [idle_cost, travelling_time_cost, bonuses_cost]
        reward = self.reward_func(time_mat, bonuses, nodes_actions)
        # print("Cost is: ", cost)

        last_state_demand = self.env_config["node_demand"][time_step-1]
        # next_state = {
        #     "idle_drivers": np.maximum(0, np.sum(nodes_actions, axis=0)
        #                                   +self.cur_state["upcoming_cars"]
        #                                   -self.cur_state["demands"]),  # [num_nodes*1] with minimum 0 since there could not be minus drivers
        #     "upcoming_cars": (self.env_config["upcoming_cars"][time_step]+
        #                      np.round(np.dot(self.env_config["demand_distribution"][time_step-1].T, last_state_demand))).astype(int), 
        #                     # upcoming consists of: newly added cars, and demands
        #     "demands": self.env_config["node_demand"][time_step], 
        #     "edge_traffic": self.env_config["edge_traffic"][time_step],
        #     "len_mat": self.len_mat, 
        #     "time_step": time_step,
        # }

        # --------------------------------------------------------------------------------------------------------------------------------------------
        #   TODO: Made the arrival cars would not minus the demands. And the upcoming cars would be zero all the times. 
        # --------------------------------------------------------------------------------------------------------------------------------------------

        next_state = {
            "idle_drivers": np.maximum(0, np.sum(nodes_actions, axis=0)),  # [num_nodes*1] with minimum 0 since there could not be minus drivers
            "upcoming_cars": np.zeros_like(last_state_demand), 
                            # upcoming consists of: newly added cars, and demands
            "demands": self.env_config["node_demand"][time_step], 
            "edge_traffic": self.env_config["edge_traffic"][time_step],
            "len_mat": self.len_mat, 
            "time_step": time_step,
        }
        self.cur_state = next_state

        return next_state, reward, done, [nodes_actions]    # next_state, reward_list [3 types of cost], done, info (information required for some algorithms)

    @ staticmethod
    def smooth_indicator(x, center=1, alpha=100):
        """
        Smooth version of the indicator function \mathbf{1}(x==1).
        Exponentially decays as x becomes farther away from 1.
        
        Parameters:
        x : float or np.ndarray
            The input value(s).
        alpha : float
            The parameter controlling the degree of smoothness.
            
        Returns:
        float or np.ndarray
            The smooth indicator value(s).
        """
        return np.exp(-alpha*(x-center)**2)

    def reward_func(self, time_mat, bonuses, nodes_actions):
        """ The reward function of MDP
        INPUT:      time_mat: ([num_nodes, num_nodes], ndarray) the travelling time between nodes 
                    bonuses: ([num_nodes, 1], ndarra) the bonuses assigned
                    nodes_actions: ([num_nodes, num_nodes], ndarray) the drivers' re-position matrix
        OUTPUT:     reward: (scalar) the comprehensive reward
         """
        # norm_factor = {"idle_cost": 1, "travelling_cost": 0.05, "bonus_cost": 0.005}
        norm_factor = {"idle_cost": 1, "travelling_cost": 0., "bonus_cost": 0.}
        coef = 100  # Rewards are primarily smaller than 1, thus multiply by 100 to make the change more obvious

        # Calculate idle/demand cost using mse loss
        # node_cars = np.sum(nodes_actions, axis=0) + self.cur_state["upcoming_cars"]   # Tempoorarily eliminate the impact of upcoming cars. 
        node_cars = np.sum(nodes_actions, axis=0)
        nodes_distribution = node_cars / np.sum(node_cars)
        demands_distribution = self.cur_state["demands"] / np.sum(self.cur_state["demands"])
        ratio = nodes_distribution / demands_distribution
        idle_cost = 1 - np.sum( self.smooth_indicator(ratio, center=1, alpha=100) ) / self.num_nodes
        # idle_cost = np.sqrt(np.sum((nodes_distribution-demands_distribution)**2))   # MSE loss between drivers distribution and demands distribution
        # idle_cost = - np.sum( nodes_distribution * np.log(nodes_distribution/demands_distribution) ) # KL divergence between two distribution. 
        idle_cost *= norm_factor['idle_cost'] * coef

        # Calculate the travelling time cost
        travelling_nodes = nodes_actions*(time_mat!=0)  # Eliminate the drivers who stay at current nodes. 
        max_time = np.sum( np.sum(travelling_nodes, axis=1)*np.max(time_mat, axis=1) )
        min_time = 0    # All staying at current node, travelling time would be zero
        avg_travelling_cost = (np.sum(nodes_actions*time_mat)-min_time) / (max_time-min_time)
        avg_travelling_cost *= norm_factor['travelling_cost'] * coef

        # Calculate bonuses cost
        max_bonus = np.sum(nodes_actions)*self.max_bonus    # Assign max_bonus to each node
        min_bonus = np.sum(nodes_actions)*self.min_bonus   # Do not assign any bonuses
        bonuses_cost = (np.sum(nodes_actions*bonuses)-min_bonus)/(max_bonus-min_bonus)
        bonuses_cost *= norm_factor['bonus_cost'] * coef

        # Calculate comprehensive cost
        overall_cost = (idle_cost + avg_travelling_cost + bonuses_cost) # Bonuses costs are not included temporarily

        return np.array([-idle_cost, -avg_travelling_cost, -bonuses_cost, -overall_cost])       

    def get_time_mat(self, len_mat, traffic, adj_mat):
        """ This function simulate the re-position process and return the time matrix 
        INPUT:      len_mat: ([num_nodes, num_nodes] ndarray) the length matrix between nodes
                    traffic: ([num_nodes, num_nodes] ndarray) the number of background traffic (cars other than taxi) driving on edges
                    adj_mat: ([num_nodes, num_nodes] ndarray) the number of background traffic (cars other than taxi) driving on edges
        OUTPUT:     travelling time matrix: ([num_nodes, num_nodes] ndarray) the travelling time between nodes
        """
        time_mat = np.zeros([self.num_nodes, self.num_nodes])
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if traffic[i,j]<1: # If no distance between i and j
                    time_mat[i,j] = 0
                else:
                    time_mat[i,j] = traffic[i,j]/100.0*len_mat[i,j]/100.0 
        return time_mat

    def get_nodes_actions(self, step):
        """ Returns the nodes_actions currently """
        return self.games[step].get_nodes_actions()
    
    def get_init_setting(self):
        """ Return init settings """
        return {"initial_drivers": self.env_config["initial_drivers"], 
                 "upcoming_cars": self.env_config["upcoming_cars"], 
                 "demands": self.env_config["node_demand"], 
                 "edge_traffic": self.env_config["edge_traffic"]}
