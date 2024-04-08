import numpy as np
import matplotlib.pyplot as plt
import copy

def normalize(vec):
    if len(vec.shape)==1:
        return vec / np.sum(vec)
    else:
        norm_vec = copy.deepcopy(vec)
        for idx, v in enumerate(vec):
            norm_vec[idx] = v / np.sum(v)
        return norm_vec

def plot_agents(T, a1_traj, a2_traj):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[6,3])
    ax1 = axs[0]
    ax1.set_ylim((0,1))
    ax1.plot(np.arange(T), np.array(a1_traj)[:, 0], label="coop", color='r')
    ax1.plot(np.arange(T), np.array(a1_traj)[:, 1], label="betray", color='b')
    ax1.set_title("Strategy for agent 1")
    ax1.legend()

    ax2 = axs[1]
    ax2.set_ylim((0,1))
    ax2.plot(np.arange(T), np.array(a2_traj)[:, 0], label="coop", color='r')
    ax2.plot(np.arange(T), np.array(a2_traj)[:, 1], label="betray", color='b')
    ax2.set_title("Strategy for agent 2")
    ax2.legend()

def plot_mdp(T, a1_traj, a2_traj, incentive_traj, V_vec_traj, update_term_traj):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[16,8])
    dim_actions = incentive_traj[-1].shape[1]

    ax1 = axs[0][0]
    ax1.plot(np.arange(T), np.array(a1_traj)[:, 0, 0], label="coop", color='r')
    ax1.plot(np.arange(T), np.array(a1_traj)[:, 0, 1], label="betray", color='b')
    ax1.set_title("Agent 1 at COOP")
    ax1.legend()

    ax2 = axs[0][1]
    ax2.plot(np.arange(T), np.array(a1_traj)[:, 1, 0], label="coop", color='r')
    ax2.plot(np.arange(T), np.array(a1_traj)[:, 1, 1], label="betray", color='b')
    ax2.set_title("Agent 1 at BETRAY")
    ax2.legend()

    ax3 = axs[0][2]
    ax3.plot(np.arange(T), np.array(a2_traj)[:, 0, 0], label="coop", color='r')
    ax3.plot(np.arange(T), np.array(a2_traj)[:, 0, 1], label="betray", color='b')
    ax3.set_title("Agent 2 at COOP")
    ax3.legend()

    ax4 = axs[0][3]
    ax4.plot(np.arange(T), np.array(a2_traj)[:, 1, 0], label="coop", color='r')
    ax4.plot(np.arange(T), np.array(a2_traj)[:, 1, 1], label="betray", color='b')
    ax4.set_title("Agent 2 at BETRAY")
    ax4.legend()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    ax5 = axs[1][0]
    ax5.plot(np.arange(T), incentive_traj[:, 0, 0], label="agent1 [c]")
    ax5.plot(np.arange(T), incentive_traj[:, 0, 1], label="agent2 [c]")
    ax5.set_title("Incentive for state COOP")
    ax5.legend()

    ax6 = axs[1][1]
    ax6.plot(np.arange(T), incentive_traj[:, 1, 0], label="agent1 [c]")
    ax6.plot(np.arange(T), incentive_traj[:, 1, 1], label="agent2 [c]")
    ax6.set_title("Incentive for state BETRAY")
    ax6.legend()

    ax7 = axs[1][2]
    ax7.plot(np.arange(T), V_vec_traj[:, 0], label="COOP", color=colors[0])
    ax7.plot(np.arange(T), V_vec_traj[:, 1], label="BETRAY", color=colors[1])
    ax7.set_title("V value traj")
    ax7.legend()

    ax8 = axs[1][3]
    for s in [0,1]:
        ax8.plot(np.arange(T), update_term_traj[:, s, 0], color=colors[2*s], label="s{} a1".format(s))
        ax8.plot(np.arange(T), update_term_traj[:, s, 1], color=colors[2*s+1], label="s{} a2".format(s))
    ax8.legend()


    # plt.plot()