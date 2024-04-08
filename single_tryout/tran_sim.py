""" This file simulate the Nash equilibrium of drivers given fixed incentive policy  """

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import numba as nb

STO_MUL_T = 1
MIN_LR = 5e-5
lr_decre = 0.5


def plot_traj(adj_mat, stratgies_traj, bonuses_traj, obj_traj, row=1, is_stochastic=False):
    n_node = adj_mat.shape[0]
    neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]

    # Plot the simulation
    nodes_color = ['k', 'b', 'r', 'y', 'c']   # color of lines
    cnt = 0
    for i in range(n_node):     # Plot the policies trajectory
        plt.subplot(2,6,1+i+(row-1)*6)
        for neighbour in neighbour_list[i]:
            plt.plot(np.arange(T), stratgies_traj[cnt, :], color=nodes_color[neighbour], label='node {}'.format(neighbour))
            if is_stochastic:
                plt.annotate("%.2f" %stratgies_traj[cnt, -1], xy=(T, stratgies_traj[cnt,-1]), xytext=(T-1000*STO_MUL_T, stratgies_traj[cnt,-1]+0.01))
            else:
                plt.annotate("%.2f" %stratgies_traj[cnt, -1], xy=(T, stratgies_traj[cnt,-1]), xytext=(T-1000, stratgies_traj[cnt,-1]+0.01))
            cnt += 1

    ax=plt.subplot(2,6,6+(row-1)*6)
    ax.set_title("bonuses")
    for i in range(n_node):     # Plot bonuses trajectory
        plt.plot(np.arange(T), bonuses_traj[i, :], color=nodes_color[i], label='node {}'.format(i))
    plt.legend()


# @nb.jit()
def simulate(demands, idle_drivers, adj_matrix, time_mat, T, lr_alpha, ID):
    # Since when incentive policy is stochastic, the convergence is really slowly, thus we increase the T to 10T
    if ID.is_stochastic: T=STO_MUL_T*T

    # Environment settings: demands and idle drivers at each nodes. 
    n_node = adj_matrix.shape[0]
    neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]
    n_neighbour = np.array([neighbour.shape[0] for neighbour in neighbour_list])

    # initialize the strategies of agents and bonuses
    strategies = [np.ones(n_neighbour[i])/n_neighbour[i] for i in range(n_node)]
    dim_strategies = np.sum([len(neighbour) for neighbour in neighbour_list])
    
    # Simulation loop
    strategies_traj = np.zeros([dim_strategies, T])
    bonuses_traj = np.zeros([n_node, T])
    ratio_traj = np.zeros([n_node, T])
    obj_traj = np.zeros(T)
    for t in trange(T):
        # Calculate the payoff
        factor = {"idle": 1, "time": 0.1, "bonuses": 0.25, 'entropy': 0.2}

        # Calculate the idle_cost
        neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]
        result = np.zeros([n_node, n_node])
        for i in range(n_node):
            result[i][neighbour_list[i]] += idle_drivers[i] * strategies[i]
        result = np.sum(result, axis=0)
        idle_cost = result / demands
        idle_stack = np.vstack([idle_cost]*n_node)

        # Calculate the travelling cost
        time_cost = np.zeros([n_node, n_node])
        for i in range(n_node):
            time_cost[i] = (time_mat[i]-np.min(time_mat[i])) / (np.max(time_mat[i])-np.min(time_mat[i]))
        
        # calculate the bonuses
        bonuses = ID.choose_bonus()
        bonuses_stack = np.vstack([bonuses]*n_node)

        # calculate the entropy cost
        entropy_cost = np.zeros([n_node, n_node])
        for i in range(n_node):
            entropy_cost[i][neighbour_list[i]] = np.log(strategies[i])

        # calculate the 
        payoff = - factor["idle"] * idle_stack - factor['time'] * time_cost + factor["bonuses"] * bonuses_stack - factor['entropy'] * entropy_cost
        ratio_traj[:, t] = idle_cost
        
        # Update lower-level agents policies 
        for i in range(n_node):
            normalizetion_factor = np.sum(  strategies[i] * np.exp(lr_alpha*payoff[i][neighbour_list[i]])  )
            start_decre_iter = 1
            if ID.is_stochastic and t>start_decre_iter:
                lr = np.maximum(lr_alpha * (t-start_decre_iter)**(-lr_decre), MIN_LR)
                strategies[i] = strategies[i] * np.exp(lr*payoff[i][neighbour_list[i]]) / normalizetion_factor
            else:
                strategies[i] = strategies[i] * np.exp(lr_alpha*payoff[i][neighbour_list[i]]) / normalizetion_factor
        strategies_traj[:, t] = np.concatenate([p for p in strategies])

        bonuses_traj[:, t] = bonuses
        obj_traj[t] = np.sum(idle_cost)
    return strategies_traj, ratio_traj, bonuses_traj, obj_traj

class IncentiveDesigner:
    """ Return designed incentve """
    def __init__(self, n_node, is_stochastic=True) -> None:
        self.n_node = n_node
        self.is_stochastic = is_stochastic
        self.incentive = np.random.uniform(0,3,5)
        self.stoc_incentive = np.zeros([n_node,4])  # each node has possibility of four bonuses [0,1,2,3]
        for i in range(n_node):
            prob = [1-self.incentive[i]/3, 0, 0, self.incentive[i]/3]
            self.stoc_incentive[i] = np.array(prob)
    
    def choose_bonus(self):
        if self.is_stochastic:
            incentive = np.zeros(self.n_node)
            for i in range(self.n_node):
                incentive[i] = np.random.choice(np.array([0,1,2,3]), size=1, p=self.stoc_incentive[i])
        else:
            incentive = self.incentive
        return incentive


if __name__ == "__main__":
    # Environment settings
    # demands = np.array([200,110,130,340,250])
    # idle_drivers = np.array([271,257,280,285,137])
    # np.random.seed(15)
    # demands = np.random.uniform(0.3, 1, [5]) * 1000
    # demands = demands / np.sum(demands) * 1000
    # idle_drivers = np.random.uniform(0.3, 1, [5]) * 1000
    # idle_drivers = idle_drivers / np.sum(idle_drivers) * 1000
    demands = np.array([[330.76929478, 157.29862111, 125.05191098, 204.5923836 , 182.28778954]])
    idle_drivers = np.array([262.9904903 , 201.51229704, 201.11597336, 148.23837386, 186.14286545])

    adj_matrix = np.array([[1, 1, 0, 1, 1],
                           [1, 1, 1, 0, 1],
                           [0, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1]])    # if i -> j is fessible, then adj_mat_{i,j} = 1
    time_mat = np.random.random([5,5]) * 1000 * (adj_matrix - np.eye(5))

    # Hyper-parameters settings
    lr_alpha = 5e-3 # learning rate for lower-level agents
    T = 5000        # Total simulation step

    plt.figure(figsize=[16,8])
    ID = IncentiveDesigner(n_node=5, is_stochastic=True)
    stratgies_traj, ratio_traj, bonuses_traj, obj_traj = simulate(demands, idle_drivers, adj_matrix, time_mat,
                                                                 T, lr_alpha, ID)
    plot_traj(adj_matrix, stratgies_traj, bonuses_traj, obj_traj, row=1, is_stochastic=True)

    ID.is_stochastic=False
    stratgies_traj, ratio_traj, bonuses_traj, obj_traj = simulate(demands, idle_drivers, adj_matrix, time_mat,
                                                                T, lr_alpha, ID)
    plot_traj(adj_matrix, stratgies_traj, bonuses_traj, obj_traj, row=2, is_stochastic=False)
    
    # plt.savefig("Comparation_result")
    plt.show()