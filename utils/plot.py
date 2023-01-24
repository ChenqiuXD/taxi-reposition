import pickle
import matplotlib.pyplot as plt
import numpy as np

import platform
import os, sys

def get_edge_index(edge_traffic):
    edge_index = []
    num_nodes = edge_traffic.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i==j:
                edge_index.append([i,j])
            else:
                if edge_traffic[i,j]!=0:
                    edge_index.append([i,j])
    return np.array(edge_index).T

def plot_result(all_args, result):
    # ------------------------------------------------------------------------------------------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # PLEASE CHANGE PLOT_TYPE before plotting. 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------------------------------------
    # Load result
    output_path = all_args.output_path
    num_step = 0    # The number of step to plot in episodes

    plot_type = {
        "idle_drivers": 1,
        "cost_traj": 1,
        "travelling_time": 0
    }

    # Get some arg parameters
    episode_length = result["init_setting"]["upcoming_cars"].shape[0]
    # Obtain current episode numbers 
    reward_traj = result['reward_traj']
    result_length = np.where( reward_traj[:,:,-1]==reward_traj[-1,:,-1] )[0][0]

    num_nodes = result["init_setting"]["upcoming_cars"].shape[1]
    edge_index = get_edge_index(result['init_setting']['edge_traffic'][0])
    colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm']

    # Plot three sequence :
    #   1 idel_drivers at next time compared to demands at next time
    #   2 sequence of rewards (four lines: idle cost, travelling time cost, bonuses cost, comprehensive cost) 
    #   3 nodes' value table

    # -------------------------------------------------------------------------------------------------------
    # 1 - idle_drivers compared to demands
    # -------------------------------------------------------------------------------------------------------
    num_avg = 2
    if plot_type["idle_drivers"]:
        fig = plt.figure(figsize=[35,10])

        # calculate the ratio between idle drivers and demands
        # data_demands = [[ result[episode_idx][num_step]["obs"]["demands"][i] for i in range(num_nodes) ] for episode_idx in range(result_length)]
        # data_idle_drivers = [[ (np.sum(result[episode_idx][num_step]["nodes actions"], axis=0)+
        #                     result[episode_idx][num_step]["obs"]["upcoming_cars"])[i]  for i in range(num_nodes) ] for episode_idx in range(result_length) ]
        # ratio = (data_demands+data_idle_drivers) / data_demands
        data_demands = np.array( [result['init_setting']['demands'][num_step]]*result_length )
        data_idle_drivers = np.array([ result["idle_drivers_traj"][episode_idx][num_step] for episode_idx in range(result_length) ])
        ratio = data_idle_drivers / data_demands

        # Plot result for each node
        for idx in range(num_nodes):
            num_subplot = 250+idx+1   # 231, 232 ... 235 for each subplot
            plt.subplot(num_subplot)
            plt.plot(np.arange(result_length-num_avg), [np.mean(ratio[i:i+num_avg,idx]) for i in range(result_length-num_avg)],
                    label="at node %d"%idx, color=colors[idx])
            
            plt.xlabel("time")
            plt.ylabel("ratio between idle and deamnds")
            plt.legend(loc="upper right")

        # Plot bonuses trajectory
        for idx in range(num_nodes):
            plt.subplot(2,5,6+idx)
            plt.plot(np.arange(result_length), 
                    np.array([ result['bonus_traj'][k][num_step][idx] for k in range(result_length) ]), 
                    label="node %d"%idx, color=colors[idx])
            plt.xlabel("episodes")
            plt.ylabel("bonuses")
            plt.legend(loc="upper right")

        plt.legend(loc="upper right")
        if platform.system()[0] == 'L':
            plt.savefig(os.path.join(output_path, 'idle_drivers.png'))
        else:
            plt.savefig(os.path.join(output_path, "idle_drivers.png"))



    # -------------------------------------------------------------------------------------------------------
    # 2- Plot sequence of reward
    # -------------------------------------------------------------------------------------------------------
    num_avg = 2
    if plot_type["cost_traj"]:
        fig_2 = plt.figure(figsize=[10,10])
        names = ["idle_reward", "travelling_reward", "bonuses_reward", "comprehensive reward"]
        num_type_reward = result['reward_traj'][0][0].shape[0]
        for i in range(num_type_reward):
            num_subplot = 220+i+1
            plt.subplot(num_subplot)
            plt.title(names[i])
            reward_sequence = [ np.mean(result['reward_traj'][j][:,i]) for j in range(result_length) ]
            plt.plot(np.arange(result_length-num_avg), np.array([np.mean(reward_sequence[j:j+num_avg]) for j in range(len(reward_sequence)-num_avg)]))
        # plt.show()
        if platform.system()[0] == 'L':
            plt.savefig(os.path.join(output_path, 'cost_traj.png'))
        else:
            plt.savefig(os.path.join(output_path, "cost_traj.png"))

    # -------------------------------------------------------------------------------------------------------
    # 3 time_mat plot
    # -------------------------------------------------------------------------------------------------------
    # if plot_type["travelling_time"]:
    #     fig = plt.figure(figsize=[15,10])
    #     for i in range(num_nodes):
    #         plt.subplot(230+i+1)
    #         for j in range(num_nodes):
    #             if j in edge_index[1, edge_index[0]==i] and j!=i:
    #                 plt.plot(np.arange(result_length), [result[k][num_step]["time mat"][i,j] for k in range(result_length) ], 
    #                         color=colors[j], label="to node %d"%j, alpha=0.5)
    #                 plt.xlabel("episodes")
    #                 plt.ylabel("travelling time")
    #                 plt.legend(loc="upper right")
    #     plt.suptitle("travelling time for idle drivers")
    #     plt.savefig(file_name[2:]+"_travelling_time.png")

    print("done")

if __name__ == "__main__":
    # TODO: Please write this function to plot steps of assigned interval
    pass