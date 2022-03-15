import numpy as np

class SumoEnv:
    """Environment of SUMO used to simulate"""
    def __init__(self, setting):
        self.num_node, self.edges, self.node_init_car, self.node_demand, \
        self.node_upcoming_car, self.node_bonus, self.edge_traffic = setting.values()

    def simulate(self, action, is_display=False):
        """Used to simulate and return the experienced traffic time
        params: 
            action: [num_node, num_node] (np.ndarray) representing the proportion of idle drivers
            is_display: Bool, whether show sumo display window
        returns: a matrix [num_node, num_node] representing the simulated travelling time with [·]_{ij} if from node i to node j
        """
        # TODO: Please complete following procedure: start simulation -> record the traffic time -> return the time matrix
        adj_mat = np.array([[1,1,0,1,1], 
                            [1,1,1,0,1], 
                            [0,1,1,1,1],
                            [1,0,1,1,1],
                            [1,1,1,1,1]])

        time_mat = np.array([[2.12526591, 0.37026576, 1.0813623 , 4.21033395, 1.02745376],
                             [4.75850159, 1.04580234, 1.44198875, 1.25536687, 0.17868027],
                             [0.41352512, 3.92683963, 2.10077262, 2.70334518, 0.7490551 ],
                             [4.60365483, 3.35208421, 4.77506637, 0.62666119, 4.23267095],
                             [3.84081488, 0.77741233, 2.37537396, 0.41995892, 0.06082066]])
        
        return np.multiply(adj_mat, time_mat)

