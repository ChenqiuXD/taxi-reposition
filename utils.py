class Node:
    # A class used to represent the node state, including all the edge features.
    def __init__(self, id, idle_drivers=100, demand=100, upcoming_cars=100, bonus=0):
        self.id = id
        self.idle_drivers = idle_drivers
        self.demand = demand
        self.upcoming_cars = upcoming_cars
        self.bonus = bonus

    def choose_action(self):
        """Choose action according to current value table"""
        pass

    def update_policy(self):
        """Update policy according to observed payoff"""


def get_adj_mat(edges):
    """Function convert COO-form edges to adajency matrix

    params: 
        edges: (np.ndarray) A COO form of connection edges. 
    return: 
        adj_mat (np.ndarray) adajency matrix
    """
    return -1
