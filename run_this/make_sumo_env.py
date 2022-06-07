import sys
import os

from torch_sparse import index_select_nnz
sys.path.append(os.getcwd())   # Added so as to use codes from rl_algo package in the parent dir

import numpy as np
from transition_model.utils import get_adj_mat, softmax, add_self_loop
from transition_model.driversGame import Game
import copy

class Arg:
    """Useless class, used to initialize Drivers' Game variable"""
    def __init__(self, lr, epoch, max_bonus, sim_steps):
        self.lr = lr
        self.epoch = epoch
        self.display = 0
        self.max_bonus = max_bonus
        self.converge_criterion = 0
        self.sim_steps = sim_steps

class Env:
    """The reinforcement learning environment of traffic simulation"""
    def __init__(self, env_config):
        self.env_config = env_config
        self.edge_index = self.env_config["edge_index"]
        self.len_mat = self.env_config["len_mat"]
        self.adj_mat = get_adj_mat(self.edge_index)
        self.n_node = self.env_config["num_node"]    # number of nodes
        self.dim_node_obs = self.env_config["dim_node_obs"]
        self.dim_edge_obs = self.env_config["dim_edge_obs"]
        self.episode_length = self.env_config["node_demand"].shape[0]
        self.max_epoch = env_config["num_env_steps"]

        self.action_space = self.env_config["dim_action"]    # action_space = 5 -> bonus range from [0,1,2,3,4]
        self.max_bonus = self.action_space-1  # 5--1 = 4
        self.obs_spaces = self.env_config["dim_node_obs"] * self.n_node + self.env_config["dim_edge_obs"] * self.edge_index.shape[1]    # node_obs * n_node + edge_obs * n_egdes

        # flag of whether reset
        self.is_reset = False
        self.lr = 0.01  # learning rate for drivers agents

        # Drivers' Game, used to simulate the payoff
        self.setting = {
            "NODE_NUM": self.n_node,  
            "EDGES": self.edge_index, 
            "NODE_INITIAL_CAR": self.env_config["initial_drivers"], # initial idle drivers  [n_node*1]
            "NODE_DEMAND": self.env_config["node_demand"][0],       # initial demands at each node  [n_node*1]
            "NODE_UPCOMING": self.env_config["upcoming_cars"][0],  # upcoming cars [n_node * 1]
            "NODE_BONUS": [0 for _ in range(self.n_node)],  # Bonuses assigned by platform agent, currently not assigned
            "EDGE_TRAFFIC" : self.env_config["edge_traffic"][0],  # edge_traffic at time 0 [n_node * n_node]
        }
        arg = Arg(lr=self.lr, epoch=self.max_epoch ,max_bonus=self.action_space-1, sim_steps=600)    # action_space is 5, max_bonus = 5-1 = 4 since bonus range from [0,1,2,3,4]
        self.games = [Game(self.setting, arg) for _ in range(self.episode_length)]    # For time_step in range(episode_length), there is a game
        self.cur_state = None # Initialize in self.reset()

    def reset(self):
        """Reset environment, initialize the observation"""
        self.is_reset = True

        # Get initial state
        init_state = {  # State format: idle_drivers, upcoming_cars, demands, edge_traffic, time_step
            "idle_drivers": self.env_config["initial_drivers"], # initial idle drivers  [n_node*1]
            "upcoming_cars": self.env_config["upcoming_cars"][0],  # upcoming cars [n_node * 1]
            "demands": self.env_config["node_demand"][0],       # initial demands at each node  [n_node*1]
            "edge_traffic": self.env_config["edge_traffic"][0],  # edge_traffic at time 0 [n_node * n_node]
            "len_mat": self.len_mat, 
            "time_step": 0
        }
        self.cur_state = init_state

        # Set initial state's variables
        self.games[0].node_init_car = init_state["idle_drivers"]
        self.games[0].node_demand = init_state["demands"]
        self.games[0].node_upcoming_car = init_state["upcoming_cars"]
        self.games[0].edge_traffic = init_state["edge_traffic"]

        return init_state

    def step(self, actions, is_warmup=False):
        """Update the states
        @params: 
            state: (dict) idle_drivers, upcoming_cars, demands, edge_traffic, time_step
            actions: (ndarray int, n_node*1) bonuses for each node, range from [0-self.action_space]
        """
        if not self.is_reset:
            raise RuntimeError("env is not reset but step function is called.")
        
        # unpack states
        time_step = self.cur_state["time_step"]
        self.games[time_step].node_init_car = self.cur_state["idle_drivers"]
        self.games[time_step].node_demand = np.maximum(self.cur_state["demands"] +\
                                            np.random.randint(low=-5, high=5, size=self.cur_state["demands"].shape), 0)
        self.games[time_step].node_upcoming_car = self.cur_state["upcoming_cars"]+\
                                            np.random.randint(low=-10, high=10, size=self.cur_state["upcoming_cars"].shape)
        self.games[time_step].edge_traffic = self.cur_state["edge_traffic"]
        self.games[time_step].node_bonus = actions
        self.games[time_step].update_node_states()

        # Get nodes' actions
        nodes_actions = (self.games[time_step].choose_actions()).astype(int)
        # To save nodes_action into recorder (in env_runner.py). Note that I assign each car directly in simulation to make sure every entry would get updated, therefore, 
        # the true nodes_actions_result should minus one to retain the original decisions of drivers. 
        self.nodes_action_result = np.maximum(0, nodes_actions-np.ones([nodes_actions.shape[0], nodes_actions.shape[1]]))
        print("Node actions are: \n", self.nodes_action_result)

        # Simulate and observe payoff
        actions_without_diagonal = copy.deepcopy(nodes_actions)  # Delete diagonal actions since they cost zero time and need not simulation
        np.fill_diagonal(actions_without_diagonal, 0)
        time_mat = self.games[time_step].simulate_game(actions_without_diagonal)
        self.sim_time_mat = time_mat

        if not is_warmup:   # While warming up, keep the policies of nodes unchanged. 
            # Calculate the payoff
            payoff = self.games[time_step].observe_payoff(time_mat, nodes_actions)

            # Update value table, return the maximum update term at value table
            self.games[time_step].update_policy(payoff, nodes_actions)

        # Prepare information of next state
        if self.cur_state["time_step"]+1 >= self.episode_length:
            self.is_reset = False
        else:
            time_step = self.cur_state["time_step"] + 1
        # time_step = self.cur_state["time_step"]+1 if self.cur_state["time_step"]+1 < self.episode_length else 0
        last_state_demand = self.env_config["node_demand"][time_step-1]
        next_state = {
            "idle_drivers": np.maximum(0, np.sum(nodes_actions, axis=0)
                                          +self.cur_state["upcoming_cars"]
                                          -self.cur_state["demands"]),  # [n_node*1] with minimum 0 since there could not be minus drivers
            "upcoming_cars": (self.env_config["upcoming_cars"][time_step]+
                             np.round(np.dot(self.env_config["demand_distribution"][time_step-1].T, last_state_demand))).astype(int), 
                            # upcoming consists of: newly added cars, and demands
            "demands": self.env_config["node_demand"][time_step], 
            "edge_traffic": self.env_config["edge_traffic"][time_step],
            "len_mat": self.len_mat, 
            "time_step": time_step,
        }
        self.cur_state = next_state

        # Calculate reward for coordinator agents: returns a list [idle_cost, travelling_time_cost, bonuses_cost]
        cost = self.calc_cost(time_mat, actions, nodes_actions) # actions are the bonuses, nodes_actions are the distribution of idle drivers
        print("Cost is: ", cost)

        return next_state, cost, False, None    # next_state, reward_list [3 types of cost], done, info

    def calc_cost(self, time_mat, bonuses, nodes_actions):
        """This function calculate the immediate cost for coordinator agent. 
        Note that the maximum cost is -10"""
        # normalization_factor = {"idle_cost": 3, "travelling_cost": 1, "bonus_cost": 1}
        normalization_factor = {"idle_cost": 3, "travelling_cost": 1.5, "bonus_cost": 1}

        # Calculate idle/demand cost using mse loss
        node_all_cars = np.sum(nodes_actions, axis=0) + self.cur_state["upcoming_cars"]
        nodes_distribution = node_all_cars / np.sum(node_all_cars)
        demands_distribution = self.cur_state["demands"] / np.sum(self.cur_state["demands"])

        # Cross Entropy
        # idle_cost = 0
        # for i in range(nodes_distribution.shape[0]):
        #     idle_cost += nodes_distribution[i]*np.log(nodes_distribution[i]/demands_distribution[i])
        # idle_cost *= 4
        
        # Idle_cost
        idle_cost = np.sqrt(np.sum((nodes_distribution-demands_distribution)**2))
        idle_cost *= normalization_factor['idle_cost']

        # Calculate the travelling time
        travelling_nodes = nodes_actions*(time_mat!=0)  # Eliminate the drivers who stay at current nodes. 
        max_time = np.sum( np.sum(travelling_nodes, axis=1)*np.max(time_mat, axis=1) )
        min_time = np.sum( np.sum(travelling_nodes, axis=1)*np.min(time_mat+(time_mat==0)*1e3, axis=1) )
        avg_travelling_cost = (np.sum(nodes_actions*time_mat)-min_time) / (max_time-min_time)
        avg_travelling_cost *= normalization_factor['travelling_cost']

        # Calculate bonuses
        bonuses_cost = np.sum(nodes_actions*bonuses)/(np.sum(nodes_actions)*self.max_bonus)
        bonuses_cost *= normalization_factor['bonus_cost']

        # Calculate comprehensive cost
        overall_cost = (0.4*idle_cost + 0.4*avg_travelling_cost + 0.2*bonuses_cost)

        return np.array([-idle_cost, -avg_travelling_cost, -bonuses_cost, -overall_cost]) * 100

    def restore(self, result):
        """This function restores the nodes' parameters
        @params:
            result: the last episode in last simulation, store in a list of dict form.
        """
        last_episode = result[-1]
        episode_length = len(last_episode)
        assert episode_length==self.episode_length  # Check whether the length of result equals episode length in self.args

        for time_step in range(episode_length):
            for i in range(self.n_node):
                self.games[time_step].nodes[i].value_table = last_episode[time_step]["nodes value"][i]

        # Restore the env_config
        self.env_config["initial_drivers"] = result[0][0]["obs"]["idle_drivers"].astype(int)
        self.env_config["node_demand"] = np.vstack([ result[0][k]["obs"]["demands"] for k in range(len(result[0])) ]).astype(int)
        self.env_config["upcoming_cars"] = np.vstack([ result[0][k]["obs"]["upcoming_cars"] for k in range(len(result[0])) ]).astype(int)
        self.env_config["edge_traffic"] = np.vstack([ result[0][k]["obs"]["edge_traffic"] for k in range(len(result[0])) ])\
                                            .astype(int).reshape(-1, self.n_node, self.n_node)

    def render(self, mode):
        """Render function"""
        pass    # Skip for now

def make_sumo_env(args):
    """This function return an env which has similar functionality as gyms. """
    edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4], 
                           [1,4,3,0,4,2,1,4,3,0,4,2,0,1,2,3]])    # COO form of connection matrix
    adj_mat_without_diagnal = get_adj_mat(edge_index)
    n_node = np.max(edge_index) + 1
    edge_index = add_self_loop(edge_index)
    adj_mat = get_adj_mat(edge_index)
    len_mat = np.zeros(edge_index.shape[1])
    for i in range(len_mat.shape[0]):
        if edge_index[0,i]==edge_index[1,i]:
            len_mat[i] = 0
        else:
            len_mat[i] = 50 if edge_index[0,i]==4 or edge_index[1,i]==4 else 70.7

    # Randomly generate some data
    # np.random.seed(10)
    episode_length = args.episode_length

    node_initial_cars = np.floor(np.random.uniform(0.2, 1, n_node) * 500).astype(int)     # Initial cars (at least 5 cars)
    node_demand = np.floor(np.random.uniform(0.3, 1, [episode_length, n_node]) * 300).astype(int)   # Demands with maximum 200
    node_distribute = get_demands_distribution(node_demand, adj_mat)
    edge_traffic = np.floor(np.random.uniform(1000, 10000, (episode_length, n_node, n_node)) * adj_mat_without_diagnal).astype(int)  # Edge traffic
    upcoming_cars = np.floor(np.random.uniform(-1, 1, [episode_length, n_node]) * 100).astype(int)   # Upcoming cars approximately 50

    env_config = {
        "num_node": n_node,
        "edge_index": edge_index, 
        "dim_node_obs": 3,  # [idle drivers, upcoming cars, demands]
        "dim_edge_obs": 2,  # [traffic flow density, length]
        "dim_action": 5,    # bonus range from [0,1,2,3,4]
        "num_env_steps": args.num_env_steps,

        "initial_drivers": node_initial_cars,   # Initial idle drivers  [n_node * 1]
        "upcoming_cars": upcoming_cars,         # Initial upcoming cars [n_node * 1]
        "node_demand": node_demand,     # Demands for each nodes, [EPISODE_LEN * n_node]
        "demand_distribution": node_distribute,     # Probability of demands distribute drivers to other nodes
        "edge_traffic": edge_traffic,   # Traffic at each edges, [EPISODE_LEN * n_node * n_node]
        "len_mat": len_mat,
    }

    return Env(env_config=env_config)

def get_demands_distribution(node_demand, adj_mat):
    """This function distribute the demands of each nodes
    @params: 
        node_demand: (ndarray, [episode_length, n_node]) demands at each node at each iteration
        adj_mat: (ndarray, [n_node, n_node]) adjcency matrix with self-loop
    """
    episode_length = node_demand.shape[0]
    n_node = adj_mat.shape[0]
    distribution = np.random.uniform(1,5,[episode_length, n_node, n_node]) * adj_mat + \
                   10000*(np.vstack([adj_mat[np.newaxis, :, :]-1]*episode_length))    # unreachable entries' values are -10000
    dist_mat = np.vstack( [ np.vstack(softmax(distribution[i][j]) for j in range(n_node)) ] for i in range(episode_length) )
    return dist_mat

