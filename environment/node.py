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
        self.num_nodes = config["num_nodes"]

        num_neighbour = len(self.neighbour_list)
        self.action_prob = 1/num_neighbour * np.ones(num_neighbour)

        self.cnt = 0

    def choose_action(self):
        """Choose action according to current value table"""        
        # Get actions
        return self.action_prob    

    def update_policy(self, payoff):
        """Update policy according to observed payoff
        INPUT:          payoff: ([num_neighbour_node, 1] ndarray) payoff experienced of neighbour nodes
        """
        self.cnt += 1
        
        # Maximise the payoff 
        normalizetion_factor = np.sum(  self.action_prob * np.exp(self.lr*payoff)  )
        self.action_prob = self.action_prob * np.exp(self.lr*payoff) / normalizetion_factor
