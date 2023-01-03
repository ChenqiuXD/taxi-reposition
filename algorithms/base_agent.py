
class BaseAgent:
    def __init__(self, args, env_config):
        """Base agent class for q-methods ('iql', 'vdn', 'qmix') agents"""
        self.args = args
        self.algo = args.algorithm_name
        self.num_nodes = env_config['num_nodes']
        self.dim_action = env_config['dim_action']
        self.dim_node_obs = env_config["dim_node_obs"]
        self.dim_edge_obs = env_config["dim_edge_obs"]
        # self.discrete_action_space = env_config['discrete_action_space']
        self.dim_obs = env_config['dim_observation']
        self.edge_index = env_config["edge_index"]
        self.len_mat = env_config["len_mat"]
        # self.episode_length = args.episode_length

        # un-tunable parameters
        self.device = self.args.device
        self.lr = self.args.lr
        self.e_greedy = self.args.epsilon
        self.batch_size = self.args.batch_size

        # Buffer
        self.buffer_size = args.buffer_size
        self.buffer = []
        self.buffer_ptr = 0

    def learn(self):
        pass

    def warmup(self):
        pass
    
    def prep_train(self):
        pass
    
    def prep_eval(self):
        pass
    
    def hard_target_update(self):
        pass

    def save_network(self):
        pass

    def append_transition(self, obs, action, reward, done, obs_, info):
        """Store transition"""
        pass

    def choose_action(self, obs, is_random=False):
        """
        Choose action according to obs.
        :param obs: observation [1, 3*self.dim_obs]
        :param is_random: (bool) whether randomly choose action
        """
        pass