import numpy as np
from transition_model.utils import softmax

class Node:
    # A class used to represent the node state, including all the edge features.
    def __init__(self, id, config, lr=1e-3, max_epoch=1000, idle_drivers=100, demand=100, upcoming_cars=100, bonus=0):
        self.id = id
        self.lr = lr
        self.max_epoch = max_epoch
        self.idle_drivers = idle_drivers
        self.demand = demand
        self.upcoming_cars = upcoming_cars
        self.bonus = bonus

        edges = config["EDGES"]
        self.neighbour_list = edges[1, np.where(edges[0]==self.id)].reshape(-1)
        self.neighbour_list = np.append(self.neighbour_list, self.id) # add self loop
        self.neighbour_list = np.unique(self.neighbour_list)    # In case sometimes the edge_index include the self-loop, added here to eliminate the repeated values
        self.num_nodes = config["NODE_NUM"]

        self.value_table = np.zeros(self.num_nodes)
        self.value_table_traj = np.zeros([self.max_epoch, self.num_nodes])

        self.cnt = 0

    def choose_action(self):
        """Choose action according to current value table"""        
        # Get actions
        action_prob = softmax(self.value_table[self.neighbour_list])
        action_cars = np.zeros(self.num_nodes)  
        for cnt, id in enumerate(self.neighbour_list):
            action_cars[id] = np.floor(action_prob[cnt] * self.idle_drivers)    # Take integer since there's no 0.5 car
        
        # Assign remaining drivers
        remain_cars = self.idle_drivers - np.sum(action_cars)
        for i in range(int(remain_cars)):
            idx = np.random.choice(self.neighbour_list, p=action_prob)
            action_cars[idx] += 1
        
        # We assign each neighbour node a car to keep its value updated. 
        action_cars[self.neighbour_list] += 1
        
        return action_cars

    def update_policy(self, payoff, actions):
        """Update policy according to observed payoff"""
        # Record the trajectory
        self.value_table_traj[self.cnt] = self.value_table
        self.cnt += 1

        # Only entries that actions is not zero would be updated
        actions_idx = (actions!=0).astype(int)

        # Update the value functions
        diff = (payoff-self.value_table) * self.lr * actions_idx
        # if self.cnt<=100:
        #     diff = (payoff-self.value_table) * self.lr * actions_idx
        # else:
        #     diff = (payoff-self.value_table) * 1/self.cnt * actions_idx
        # diff = (payoff-self.value_table) * 1/self.cnt
        self.value_table += diff

        return np.max(np.abs(diff))
