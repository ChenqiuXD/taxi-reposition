import numpy as np

class Node:
    # A class used to represent the node state, including all the edge features.
    def __init__(self, id, config, lr=1e-3, max_epoch=1000, warmup_epoch=100):
        self.id = id
        self.lr = lr
        self.max_epoch = max_epoch

        self.warmup_epoch = warmup_epoch
        self.is_warmed_up = False

        edges = config["edge_index"]
        self.neighbour_list = edges[1, np.where(edges[0]==self.id)].reshape(-1)
        self.neighbour_list = np.append(self.neighbour_list, self.id) # add self loop
        self.neighbour_list = np.unique(self.neighbour_list)    # In case sometimes the edge_index include the self-loop, added here to eliminate the repeated values
        self.num_nodes = config["num_nodes"]
        self.idle_drivers = 0

        self.value_table = np.zeros(self.num_nodes)
        # self.value_table_traj = np.zeros([self.max_epoch, self.num_nodes])

        num_neighbour = len(self.neighbour_list)
        self.action_prob = 1/num_neighbour * np.array([idx in self.neighbour_list for idx in range(self.num_nodes)])
        # self.action_prob_traj = np.zeros([self.max_epoch, self.num_nodes])

        self.cnt = 0

    def choose_action(self):
        """Choose action according to current value table"""        
        # Get actions
        # action_prob = softmax(self.value_table[self.neighbour_list])
        return self.action_prob    

    def update_policy(self, payoff):
        """Update policy according to observed payoff
        INPUT:          payoff: ([num_neighbour_node, 1] ndarray) payoff experienced of neighbour nodes
        """
        # Record the trajectory
        # self.value_table_traj[self.cnt] = self.value_table
        # self.action_prob_traj[self.cnt] = self.action_prob
        self.cnt += 1
        if not self.is_warmed_up and self.cnt>=self.warmup_epoch:   # Renew the counter to eliminate trajectory during warm up. 
            self.is_warmed_up = True
            self.cnt=0

        # Update the value functions
        diff = (payoff-self.value_table) * self.lr
        self.value_table += diff

        # Update the action functions
        val = np.exp(self.lr*payoff[self.neighbour_list])*np.exp( (1-self.lr)*np.log(self.action_prob[self.neighbour_list]) )
        self.action_prob[self.neighbour_list] = val / np.sum(val)

        return np.max(np.abs(diff))
