import pickle
import matplotlib.pyplot as plt
import numpy as np
import re

def get_edge_index(edge_traffic):
    edge_index = []
    num_node = edge_traffic.shape[0]
    for i in range(num_node):
        for j in range(num_node):
            if i==j:
                edge_index.append([i,j])
            else:
                if edge_traffic[i,j]!=0:
                    edge_index.append([i,j])
    return np.array(edge_index).T

# ------------------------------------------------------------------------------------------------
# Suggested result: 
# ./heuristic_episodes_80_length_6 - step_cnt=2
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE CHANGE PLOT_TYPE before plotting. 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------------------
plot_type = {
    "idle_drivers": 1,
    "cost_traj": 1,
    "value_table": 1,
    "travelling_time": 0
}

# Load result
file_name = "./q_learning_episodes_900_length_6_seed_11"
if file_name[2:].startswith('q'):
    algo_name = "q_learning"
else:
    algo_name = file_name[2:file_name.find('_')]
f = open(file_name, 'rb')
result = pickle.load(f)

# Assign the step you are interested in
step_cnt = 2    # range from 0-4. Although there is 6 steps in total, however in ploting 1st graph, we compare between demands[step_cnt] and idle_drivers[step_cnt+1]
episode_cnt = [0, 10, 50, 100, 200, 299]
# episode_cnt = [0, 100, 200, 300, 450, 599]
# episode_cnt = [0, 20, 50, 75, 99]

# Get some arg parameters
episode_length = len(result[0])
num_node = result[0][0]["obs"]["idle_drivers"].shape[0]
edge_index = get_edge_index(result[0][0]["obs"]["edge_traffic"])
colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm']



# Plot three sequence :
#   1 idel_drivers at next time compared to demands at next time
#   2 sequence of rewards (four lines: idle cost, travelling time cost, bonuses cost, comprehensive cost) 
#   3 nodes' value table

# -------------------------------------------------------------------------------------------------------
# 1 - idle_drivers compared to demands
# -------------------------------------------------------------------------------------------------------
if plot_type["idle_drivers"]:
    num_episode = len(episode_cnt)
    fig = plt.figure(figsize=[15,10])
    for idx, episode_idx in enumerate(episode_cnt):
        num_subplot = 230+idx+1   # 231, 232 ... 235 for each subplot
        plt.subplot(num_subplot)
        plt.title("%dth episode"%episode_idx)
        data_demands = [ result[episode_idx][step_cnt]["obs"]["demands"][i] for i in range(num_node) ]
        data_idle_drivers = [ (np.sum(result[episode_idx][step_cnt]["nodes actions"], axis=0)+
                                result[episode_idx][step_cnt]["obs"]["upcoming_cars"])[i]  for i in range(num_node) ]
        plt.plot(np.arange(num_node), data_demands, 'or-', label="demands")
        plt.bar(np.arange(num_node), data_idle_drivers, alpha=0.3, color='blue', label="idle_drivers")
        plt.xlabel("time")
        plt.ylabel("num of drivers")

        for a,b in zip(np.arange(num_node), data_demands):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
        for a,b in zip(np.arange(num_node), data_idle_drivers):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper right")

    plt.suptitle("Idle drivers at {} step and demands at {} step".format(step_cnt, step_cnt-1), fontsize='xx-large')
    # plt.legend(loc="upper right")
    # plt.show()
    plt.savefig(file_name[2:]+"_idle_drivers.png")



# -------------------------------------------------------------------------------------------------------
# 2- Plot sequence of reward
# -------------------------------------------------------------------------------------------------------
if plot_type["cost_traj"]:
    fig_2 = plt.figure(figsize=[10,10])
    names = ["idle_cost", "travelling_cost", "bonuses_cost", "comprehensive cost"]
    num_type_reward = len(result[0][0]["reward"])
    reward_sequence = np.zeros([len(result), num_type_reward])
    # for i in range(len(result)):
    #     for j in range(episode_length):
    #         reward_sequence[i] += result[i][j]["reward"]
    #     reward_sequence[i] = reward_sequence[i]*1.0/episode_length  # Episode average cost. 
    for i in range(num_type_reward):
        num_subplot = 220+i+1
        plt.subplot(num_subplot)
        plt.title(names[i])
        # plt.plot(np.arange(len(result)), reward_sequence[:, i])
        plt.plot(np.arange(len(result)), [result[j][step_cnt]['reward'][i] for j in range(len(result))])
    # plt.show()
    plt.savefig(file_name[2:]+"_cost_traj.png")



# -------------------------------------------------------------------------------------------------------
# 3 node's value table
# -------------------------------------------------------------------------------------------------------
if plot_type["value_table"]:
    plt.figure(figsize=[35, 15])
    x = range(len(result))
    for i in range(num_node):
        plt.subplot(3,5,i+1)
        for j in range(num_node):
            if j in edge_index[1, edge_index[0]==i]:
                plt.plot(np.arange(len(result)), 
                        np.array([ result[k][step_cnt]['nodes value'][i, j] for k in range(len(result))]),
                        label="to node %d"%j, color=colors[j])
        plt.xlabel("episodes")                 
        plt.ylabel("value table")
        plt.legend(loc="upper right")

    # Plot nodes' actions
    x = range(len(result))
    for i in range(num_node):
        plt.subplot(3,5,5+i+1)
        for j in range(num_node):
            if j in edge_index[1, edge_index[0]==i]:
                plt.plot(np.arange(len(result)), 
                        np.array([ result[k][step_cnt]['nodes actions'][i, j] for k in range(len(result))]), 
                        label="to node %d"%j, color=colors[j])
        plt.xlabel("episodes")                 
        plt.ylabel("nodes'actions")
        plt.legend(loc="upper right")

    # Plot the trend of idle drivers
    plt.subplot(3,5,11)
    for i in range(num_node):
        plt.plot(np.arange(len(result)),
                # obs at step_cnt+1 could represent the effect of nodes actions at step_cnt
                np.array([ (result[k][step_cnt]['obs']['upcoming_cars']+
                            np.sum(result[k][step_cnt]["nodes actions"], axis=0)-
                            result[k][step_cnt]['obs']["demands"])[i] for k in range(len(result)) ]), 
                label="node %d"%i, color=colors[i])

        plt.plot(np.arange(len(result)), 
                np.array([ result[k][step_cnt]['obs']['demands'][i] for k in range(len(result)) ]), 
                label="demands %d"%i, color=colors[i], linestyle='--', alpha=0.5)
        
        plt.xlabel("episodes")
        plt.ylabel("num of idle drivers")
        plt.legend(loc="upper right")

    # Plot the trend of bonuses
    plt.subplot(3,5,12)
    for i in range(num_node):
        plt.plot(np.arange(len(result)), 
                np.array([ result[k][step_cnt]["action"][i] for k in range(len(result)) ]), 
                label="node %d"%i, color=colors[i])
        plt.xlabel("episodes")
        plt.ylabel("bonuses")
        plt.legend(loc="upper right")

    # Plot the frame of original data
    plt.subplot(3,5,13)
    cols = ["node {}".format(str(i)) for i in range(num_node)]
    rows = ["demands", "upcoming cars", "traffic"]
    # Obtain data
    data = [[0]*num_node] * len(rows)
    # data[0] = list(self.node_init_car)
    data[0] = list(result[0][step_cnt]['obs']["demands"])
    data[1] = list(result[0][step_cnt]['obs']["upcoming_cars"])
    traffic = list(re.sub(' +', ' ', str(result[0][step_cnt]['obs']["edge_traffic"][i]))[1:-1] for i in range(num_node))
    traffic = [re.sub(' ', '\n', traffic[i]).lstrip().rstrip() for i in range(num_node)]
    data[2] = traffic
    tab = plt.table(cellText=data, colLabels=cols, rowLabels=rows, loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)    

    # Assign the height of last rows
    cell_dict = tab.get_celld()
    for i in range(len(cols)):
        cell_dict[(len(rows),i)]._height = 0.4
        for j in range(len(rows)+1):
            cell_dict[(j, i)]._width = 0.15

    plt.suptitle("value table, nodes actions at step {} and idle_drivers at step {}".format(step_cnt,step_cnt+1), fontsize='xx-large')
    plt.axis('tight')
    plt.axis('off')
    plt.savefig(file_name[2:]+"_value_table_traj.png")



# -------------------------------------------------------------------------------------------------------
# 4 time_mat plot
# -------------------------------------------------------------------------------------------------------
if plot_type["travelling_time"]:
    fig = plt.figure(figsize=[15,10])
    for i in range(num_node):
        plt.subplot(230+i+1)
        for j in range(num_node):
            if j in edge_index[1, edge_index[0]==i] and j!=i:
                plt.plot(np.arange(len(result)), [result[k][step_cnt]["time mat"][i,j] for k in range(len(result)) ], 
                        color=colors[j], label="to node %d"%j, alpha=0.5)
                plt.xlabel("episodes")
                plt.ylabel("travelling time")
                plt.legend(loc="upper right")
    plt.suptitle("travelling time for idle drivers")
    plt.savefig(file_name[2:]+"_travelling_time.png")

print("done")
