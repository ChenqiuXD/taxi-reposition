from torch.utils.tensorboard import SummaryWriter
import os

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

        # Tensorboard writer
        self.train_steps = 0
        self.writer = SummaryWriter(log_dir=args.output_path)

    def learn(self):
        """ Agent learns to adapt the bonuses policy """
        pass

    def warmup(self):
        """ Agents collect transitions to fill the buffer """
        pass
    
    def prep_train(self):
        """ Call net.train() to prepare the network for training """
        pass
    
    def prep_eval(self):
        """ Call net.eval() to prepare the network for evaluation """
        pass
    
    def hard_target_update(self):
        """ Copying the target network from training network """
        pass

    def save_model(self, output):
        """ Save neural network """
        pass

    def load_model(self, output):
        """ Load neural network """
        pass

    def append_transition(self, obs, action, reward, done, obs_, info):
        """ Store transition """
        raise NotImplementedError("The agent does not implement the append_transition method. ")

    def choose_action(self, obs, is_random=False):
        """
        Choose action according to obs.
        :param obs: observation [1, 3*self.dim_obs]
        :param is_random: (bool) whether randomly choose action

        :Output: action range from [-1,1]
        """
        raise NotImplementedError("The agent does not implement the choose_action method. ")