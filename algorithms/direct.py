import numpy as np
from algorithms.base_agent import BaseAgent
from environment.utils import get_adj_mat

class DirectPolicy(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        self.env_config = env_config
        self.episode_length = args.episode_length

        self.num_nodes = env_config["num_nodes"]
        self.idle_drivers_list = np.zeros([ self.episode_length, self.num_nodes ])
        self.demands_list = np.zeros([ self.episode_length, self.num_nodes ])
        self.nodes_actions_list = np.zeros([ self.episode_length, self.num_nodes, self.num_nodes ])

        edge_index = env_config['edge_index']
        self.adj_mat = get_adj_mat(edge_index)
        self.neighbour_list = [ np.where(line==1)[0] for line in self.adj_mat ]
        self.num_neighbour = [ np.sum(line).astype(int) for line in self.adj_mat ]
        self.min_bonus = args.min_bonus
        self.max_bonus = args.max_bonus

        self.action = np.ones([self.episode_length, self.num_nodes])*(self.min_bonus)   # start with minimum bonuses
        self.min_bonus = args.min_bonus
        self.max_bonus = args.max_bonus

        self.A = np.zeros([self.num_nodes, np.sum(self.num_neighbour)])
        cnt = 0
        for i in range(self.num_nodes):
            self.A[ i, cnt: cnt+self.num_neighbour[i] ] = np.ones(self.num_neighbour[i])
            cnt += self.num_neighbour[i]
        self.nabla_y_F = np.concatenate([np.eye(self.num_nodes)[self.neighbour_list[i]].T for i in range(self.num_nodes)], axis=1)

        # calculate the S matrix
        self.S = []
        for i in range(self.num_nodes):
            tmp = np.zeros([self.num_nodes, self.num_nodes])
            for j in range(self.num_nodes):
                if j in self.neighbour_list[i]:
                    tmp[j, j] += 1
            self.S.append(tmp)

        self.cur_time = 0

    def append_transition(self, obs, action, reward, done, obs_, info):
        time_step = obs["time_step"]
        self.cur_time = time_step
        self.idle_drivers_list[time_step] = obs["idle_drivers"]
        self.demands_list[time_step] = obs["demands"]
        self.nodes_actions_list[time_step] = info[0]

    def learn(self):
        """ Change bonuses according to recorded idle_drivers/demands ratios. Note that the range of bonus are [-1, 1]"""
        time_step = self.cur_time
        policies = np.concatenate( [ (node_policy/np.sum(node_policy))[self.neighbour_list[i]]\
                                    for i, node_policy in enumerate(self.nodes_actions_list[time_step])  ] )
        L = self.get_L( policies, self.idle_drivers_list[time_step], self.demands_list[time_step] )
        M = L - L @ self.A.T @ np.linalg.inv(self.A@L@self.A.T) @ self.A @ L
        nabla_y_x = (-M@self.nabla_y_F.T).T
        update_term = np.zeros(self.num_nodes)

        index = np.zeros(self.num_nodes).astype(int)
        cnt = 0
        for i in range(self.num_nodes):
            index[i] = int(cnt)
            cnt += self.num_neighbour[i] 

        cars = np.sum( self.nodes_actions_list[time_step], axis=0 )
        cars_distribution = cars / np.sum(cars)
        demands_distribution = self.demands_list[time_step] / np.sum(self.demands_list[time_step])

        # demands distribution lies in the front in the KL divergence between drivers and demands
        for i in range(self.num_nodes):
            for j in self.neighbour_list[i]:
                idx = np.where(i==self.neighbour_list[j])[0][0]
                update_term += ( np.log(cars_distribution[i]) + 1 -np.log(demands_distribution[i]) ) * cars_distribution[j] * nabla_y_x[:, index[j]+idx]

        self.action[time_step] -= self.lr * update_term
        self.action[time_step] = np.maximum(self.min_bonus, self.action[time_step])
        self.action[time_step] = np.minimum(self.max_bonus, self.action[time_step])

    def choose_action(self, obs, is_random=False):
        """ return actions, range from [-1,1] """
        if is_random:
            return np.random.random([self.num_nodes])*( self.max_bonus -self.min_bonus ) + self.min_bonus
        else:
            time_step = obs["time_step"]
            return self.action[time_step]
    
    def get_L(self, policies, idle_drivers, demands):
        """ This function calculate the L matrix """
        n_node = idle_drivers.shape[0]
        dim_policies = np.sum(self.num_neighbour)

        L = np.zeros([dim_policies, dim_policies])
        cnt_col = 0
        for i in range(n_node):
            cnt_row = 0
            for j in range(n_node):
                sub_mat = (1+(i==j)) * idle_drivers[i] * self.S[j][self.neighbour_list[j]][:, self.neighbour_list[i]]
                L[cnt_row:cnt_row+self.num_neighbour[j], cnt_col:cnt_col+self.num_neighbour[i]] = sub_mat
                cnt_row += self.num_neighbour[j]
            for idx, neighbour_node in enumerate(self.neighbour_list[i]):
                L[:, cnt_col+idx:cnt_col+idx+1] /= demands[neighbour_node]            
            cnt_col += self.num_neighbour[i]

        # Entropy would change the diagonal of L
        L += np.diag(1/policies)
        return -L
