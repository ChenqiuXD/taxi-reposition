""" This file is a single episode test environment for the Hong's algorithm: inducing desired equilibrium using incentive """

import numpy as np
import matplotlib.pyplot as plt



def get_L(policies, adj_matrix, idle_drivers, demands):
    """ This function calculate the L matrix """
    n_node = idle_drivers.shape[0]
    neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]
    n_neighbour = np.array([neighbour.shape[0] for neighbour in neighbour_list])
    dim_policies = np.sum(n_neighbour)

    # calculate the S matrix
    S = []
    for i in range(n_node):
        tmp = np.zeros([n_node, n_node])
        for j in range(n_node):
            if j in neighbour_list[i]:
                tmp[j, j] += 1
        S.append(tmp)

    L = np.zeros([dim_policies, dim_policies])
    cnt_col = 0
    for i in range(n_node):
        cnt_row = 0
        for j in range(n_node):
            sub_mat = (1+(i==j)) * idle_drivers[i] * S[j][neighbour_list[j]][:, neighbour_list[i]]
            L[cnt_row:cnt_row+n_neighbour[j], cnt_col:cnt_col+n_neighbour[i]] = sub_mat
            cnt_row += n_neighbour[j]
        for idx, neighbour_node in enumerate(neighbour_list[i]):
            L[:, cnt_col+idx:cnt_col+idx+1] /= demands[neighbour_node]            
        cnt_col += n_neighbour[i]

    # Entropy would change the diagonal of L
    L += np.diag(1/np.concatenate(policies))
    return -L

def plot_traj(adj_mat, policies_traj, ratio_traj, bonuses_traj, obj_traj):
    n_node = adj_mat.shape[0]
    neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]

    # Plot the simulation
    plt.figure(figsize=[14,10])
    nodes_color = ['k', 'b', 'r', 'y', 'c']   # color of lines
    cnt = 0
    for i in range(n_node):     # Plot the policies trajectory
        plt.subplot(251+i)
        for neighbour in neighbour_list[i]:
            plt.plot(np.arange(T*T_inner), policies_traj[cnt, :], color=nodes_color[neighbour], label='node {}'.format(neighbour))
            cnt += 1
        plt.legend()

    ax=plt.subplot(256)
    ax.set_title("bonuses")
    for i in range(n_node):     # Plot bonuses trajectory
        plt.plot(np.arange(T), bonuses_traj[i, :], color=nodes_color[i], label='node {}'.format(i))
    plt.legend()

    ax = plt.subplot(257)
    ax.set_title("idle drivers")
    plt.bar(np.arange(n_node), idle_drivers, color=nodes_color)
    plt.xticks(np.arange(n_node), [str(i) for i in range(n_node)])

    ax = plt.subplot(258)
    ax.set_title("demands")
    plt.bar(np.arange(n_node), demands, color=nodes_color)
    plt.xticks(np.arange(n_node), [str(i) for i in range(n_node)])

    ax = plt.subplot(259)
    ax.set_title("ratio trajectory")
    for i in range(n_node):
        plt.plot(np.arange(T*T_inner), ratio_traj[i,:], color=nodes_color[i], label="node {}".format(i))
    plt.legend()

    ax = plt.subplot(2,5,10)
    ax.set_title("objective trajectory")
    plt.plot(np.arange(T), obj_traj[:])

    plt.show()

def simulate(MIN_BONUS, MAX_BONUS, demands, idle_drivers, adj_matrix, time_mat, T, T_inner, lr_alpha, lr_beta, policy_type):
    # Environment settings: demands and idle drivers at each nodes. 
    n_node = demands.shape[0]
    neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]
    n_neighbour = np.array([neighbour.shape[0] for neighbour in neighbour_list])

    # initialize the policies of agents and bonuses
    policies = [np.ones(n_neighbour[i])*(1/n_neighbour[i]) for i in range(n_node)]
    dim_policies = np.sum([len(neighbour) for neighbour in neighbour_list])
    bonuses = np.zeros(n_node)

    # sensitivity related constant
    A = np.zeros([n_node, dim_policies])    # constraint matrix: only constraint that policies sums are 1
    cnt = 0
    for i in range(n_node):
        A[i, cnt:cnt+n_neighbour[i]] = np.ones(n_neighbour[i])
        cnt += n_neighbour[i]
    nabla_y_F = np.concatenate([np.eye(n_node)[neighbour_list[i]].T for i in range(n_node)], axis=1)
    
    # Simulation loop
    policies_traj = np.zeros([dim_policies, T*T_inner])
    ratio_traj = np.zeros([n_node, T*T_inner])
    bonuses_traj = np.zeros([n_node, T])
    obj_traj = np.zeros(T)
    for t in range(T):
        # Inner loop: agnets update their policies
        for inner_t in range(T_inner):
            # Calculate the payoff
            factor = {"idle": 1, "time": 0.5, "bonuses": 0.25, 'entropy': 0.2}

            # Calculate the idle_cost
            n_node = adj_matrix.shape[0]
            neighbour_list = [np.where(adj_matrix[i]==1)[0] for i in range(n_node)]
            result = np.zeros([n_node, n_node])
            for i in range(n_node):
                result[i][neighbour_list[i]] += idle_drivers[i] * policies[i]
            result = np.sum(result, axis=0)
            idle_cost = result / demands
            idle_stack = np.vstack([idle_cost]*n_node)

            # Calculate the travelling cost
            time_cost = np.zeros([n_node, n_node])
            for i in range(n_node):
                time_cost[i] = (time_mat[i]-np.min(time_mat[i])) / (np.max(time_mat[i])-np.min(time_mat[i]))
            
            # calculate the bonuses
            bonuses_stack = np.vstack([bonuses]*n_node)

            # calculate the entropy cost
            entropy_cost = np.zeros([n_node, n_node])
            for i in range(n_node):
                entropy_cost[i][neighbour_list[i]] = np.log(policies[i])

            # calculate the 
            payoff = -factor["idle"] * idle_stack -factor['time'] * time_cost + factor["bonuses"] * bonuses_stack - factor['entropy'] * entropy_cost
            ratio_traj[:, t*T_inner+inner_t] = idle_cost
            
            # Update lower-level agents policies 
            for i in range(n_node):
                normalizetion_factor = np.sum(  policies[i] * np.exp(lr_alpha*payoff[i][neighbour_list[i]])  )
                policies[i] = policies[i] * np.exp(lr_alpha*payoff[i][neighbour_list[i]]) / normalizetion_factor
            policies_traj[:, t*T_inner+inner_t] = np.concatenate([p for p in policies])

        # Outer loop: upper-level agent update
        if policy_type == 'grad':
            L = get_L(policies, adj_matrix, idle_drivers, demands)
            M = L-L@A.T@ np.linalg.inv(A@L@A.T) @ A@L
            nabla_y_x = (-M @ nabla_y_F.T).T
            update_term = np.zeros(n_node)
            
            index = np.zeros(n_node).astype(int)
            cnt = 0
            for i in range(n_node):
                index[i] = int(cnt)
                cnt += n_neighbour[i] 

            cars = np.zeros([n_node])
            for i in range(n_node):
                cars[neighbour_list[i]] += idle_drivers[i] * policies[i]
            cars_distribution = cars / np.sum(cars)
            demands_distribution = demands / np.sum(demands)

            # demands distribution lies in the front in the KL divergence between drivers and demands
            for i in range(n_node):
                for j in neighbour_list[i]:
                    idx = np.where(i==neighbour_list[j])[0][0]
                    update_term += ( np.log(cars_distribution[i]) + 1 -np.log(demands_distribution[i]) ) * cars_distribution[j] * nabla_y_x[:, index[j]+idx]

            bonuses -= lr_beta * update_term
            bonuses = np.maximum(MIN_BONUS, bonuses)
            bonuses = np.minimum(MAX_BONUS, bonuses)
            # print("At iteration {}, bonuses are {}".format(t, bonuses))
            # print("At iteration {}, payoff are: {}\n".format(t, payoff))
        elif policy_type == 'null':
            bonuses = np.zeros(5)
        else:
            raise RuntimeError("The bonus_policy_type parameter {} does not correspond to any existing methods, please rechek.".format(policy_type))
        
        bonuses_traj[:, t] = bonuses
        obj_traj[t] = np.sum(idle_cost)
    return policies_traj, ratio_traj, bonuses_traj, obj_traj


if __name__ == "__main__":
    # Environment settings
    # demands = np.array([200,110,130,340,250])
    # idle_drivers = np.array([271,257,280,285,137])
    np.random.seed(15)
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
    MIN_BONUS = 0
    MAX_BONUS = 100
    lr_alpha = 5e-3 # learning rate for lower-level agents
    lr_beta =  1e-3    # learning rate for upper-level agent
    T = 10000        # Total simulation step
    T_inner = 1    # Lower-level agents update T_inner times, the upper agent would update once. 

    policy_type = ['grad', 'null']
    policy = 'grad'
    policies_traj, ratio_traj, bonuses_traj, obj_traj = simulate(MIN_BONUS, MAX_BONUS, demands, idle_drivers, 
                                                                adj_matrix, time_mat, T, T_inner, lr_alpha, lr_beta, policy_type=policy)

    plot_traj(adj_matrix, policies_traj, ratio_traj, bonuses_traj, obj_traj)