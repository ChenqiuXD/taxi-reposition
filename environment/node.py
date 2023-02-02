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
        # self.idle_drivers = 0

        num_neighbour = len(self.neighbour_list)
        self.action_prob = 1/num_neighbour * np.ones(num_neighbour)
        # self.action_prob_traj = np.zeros([self.max_epoch, num_neighbour])

        self.cnt = 0

    def choose_action(self):
        """Choose action according to current value table"""        
        # Get actions
        return self.action_prob    

    def update_policy(self, payoff):
        """Update policy according to observed payoff
        INPUT:          payoff: ([num_neighbour_node, 1] ndarray) payoff experienced of neighbour nodes
        """
        # Record the trajectory
        # if self.is_warmed_up:
        #     self.action_prob_traj[self.cnt] = self.action_prob
        # elif self.cnt>=self.warmup_epoch:   # Renew the counter to eliminate trajectory during warm up. We require the cnt to know whether warmup has finished. 
        #     self.is_warmed_up = True
        #     self.cnt=0

        self.cnt += 1
        
        # Maximise the payoff 
        normalizetion_factor = np.sum(  self.action_prob * np.exp(self.lr*payoff)  )
        self.action_prob = self.action_prob * np.exp(self.lr*payoff) / normalizetion_factor
