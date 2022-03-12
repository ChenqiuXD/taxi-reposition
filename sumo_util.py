
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
        pass

