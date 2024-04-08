import numpy as np
from util_functions import normalize, plot_agents, plot_mdp
import matplotlib.pyplot as plt
from tqdm import trange

utility_coop_a1 = np.array([ [2,4],[-4,-2] ])
utility_coop_a2 = np.array([ [2,4],[-4,-2] ])
utility_betray_a1 = np.array([ [2,4],[-4,-2] ])
utility_betray_a2 = np.array([ [2,4],[-4,-2] ])


def update_strategies(s, a1, a2, i1, i2, lamda, alpha):
    """ Update strategies one iteration """
    if s == 0:  # COOP
        u1 = utility_coop_a1
        u2 = utility_coop_a2
    elif s == 1:    # BETRAY
        u1 = utility_betray_a1
        u2 = utility_betray_a2
    
    # Calculate strategy before normalization
    a1_ = a1 * np.exp( alpha * (a2.T @ u1 + i1 - lamda * np.ones(2) - lamda * np.log(a1)) )
    a2_ = a2 * np.exp( alpha * (a1.T @ u2 + i2 - lamda * np.ones(2) - lamda * np.log(a2)) )

    # Normalize
    a1_normalized = normalize(a1_)
    a2_normalized = normalize(a2_)

    return a1_normalized, a2_normalized

def get_sensetivity_mat(s, a1, a2, lamda):
    """ This function return the sensitivity matrix of current game """
    if s == 0:  # COOP
        u1 = utility_coop_a1
        u2 = utility_coop_a2
    elif s == 1:    # BETRAY
        u1 = utility_betray_a1
        u2 = utility_betray_a2

    nabla_a_F = np.vstack( [ np.hstack([-lamda*np.diag(1/a1), u1]), np.hstack([u2, -lamda*np.diag(1/a2)]) ] )
    L = np.linalg.inv(nabla_a_F)

    A = np.array([[1,1,0,0], [0,0,1,1]])    # A represents the simplex constraint, i.e., the sum of a1 and a2 should be one
    M = L-L@A.T@ np.linalg.inv(A@L@A.T) @ A@L
    nabla_y_F = np.array([[1,0,0,0], [0,0,1,0]])

    nabla_y_x = (-M @ nabla_y_F.T).T
    return nabla_y_x

def get_P_vec(a1_s, a2_s):
    """ Return transition probability vector of current state """
    P = np.zeros(2)
    P[0] = a1_s[0] * a2_s[0]
    P[1] = 1-P[0]
    return P

def get_P_mat(a1, a2):
    """ Return transition probability matrix of current game strategies"""
    P = np.zeros([2,2])
    P[0] = get_P_vec(a1[0], a2[0])  # COOP -> [COOP, BETRAY]
    P[1] = get_P_vec(a1[1], a2[1])  # BETRAY -> [COOP, BETRAY]
    return P

def get_nabla_a_Q(a1, a2, gamma):
    # calculate the V value function
    R_vec = np.zeros(2)
    for s in [0,1]:
        R_vec[s] = R2(s, a1[s], a2[s])
    P_mat = get_P_mat(a1, a2)
    V_vec = np.linalg.inv(np.eye(2) - gamma*P_mat) @ R_vec
    V_vec_traj[t] = V_vec

    # Calculate nabla_a Q(s,a)
    nabla_a_Q = np.zeros([2, 4])
    nabla_a_P = np.array( [ [[a2[0][0], 0, a1[0][0], 0],     # nabla_a P(COOP|COOP,a)
                             [a2[0][1], 1, a1[0][1], 1]],     # nabla_a P(BETRAY|COOP,a)
                            [[a2[1][0], 0, a1[1][0], 0],     # nabla_a P(COOP|BETRAY,a)
                             [a2[1][1], 1, a1[1][1], 1]]] )   # nabla_a P(BETRAY|BETRAY,a)
    # nabla_a_R = np.array( [  ] )
    nabla_a_Q[0] = np.array([-2*(a1[0][0]-COOP_VAL[0]), 0, -2*(a2[0][0]-COOP_VAL[1]), 0])\
                     + gamma * ( nabla_a_P[0][0] * V_vec[0] + nabla_a_P[0][1] * V_vec[1] ) # nabla_a_R + gamma * nabla_a_P @ {V(s')}_{s'\in S}
    nabla_a_Q[1] = np.array([0, -2*(a1[1][1]-BETRAY_VAL[0]), 0, -2*(a2[1][1]-BETRAY_VAL[1])])\
                     + gamma * ( nabla_a_P[1][0] * V_vec[0] + nabla_a_P[1][1] * V_vec[1] ) # nabla_a_R + gamma * nabla_a_P @ {V(s')}_{s'\in S}

    return nabla_a_Q


def get_nash(incentive):
    a1 = np.array([[0.5,0.5],[0.5,0.5]])
    a2 = np.array([[0.5,0.5],[0.5,0.5]])
    for _ in range(50):
        for s in [0,1]:
            a1[s], a2[s] = update_strategies(s, a1[s], a2[s], [incentive[s][0], 0], [incentive[s][1], 0], lamda, lr_a)
    return a1, a2

""" Main function """
COOP_VAL = [0.65, 0.7]   # The value of cooperate action in COOP state
BETRAY_VAL = [0.8, 0.7] # The value of betray action in BETRAY state

def R2(s, a1, a2):
    assert s==0 or s==1, "The input s is {}, which is neighter 0 nor 1".format(s)
    if s==0:
        reward = 2.75 - (a1[0]-COOP_VAL[0])**2 - (a2[0]-COOP_VAL[1])**2    # COOP
    elif s==1:
        reward = 2. - (a1[1]-BETRAY_VAL[0])**2 - (a2[1]-BETRAY_VAL[1])**2   # BETRAY
    return reward

lamda = 2.5     # Temperature parameter for agent's strategy
delta = 1e-3    # Regularization term for incentive \theta
theta_max = 100
theta_min = -100
T = 1500
lr_a = 3e-1
lr_i = 1.
gamma = 0.9

a1 = np.array([[0.5,0.5],[0.5,0.5]])
a2 = np.array([[0.5,0.5],[0.5,0.5]])
ID_policy = np.zeros([2,2])   # Initial incentive is set to zero

# Record trajector
a1_traj = np.zeros([T, 2, 2])   # [n_iter, n_state, n_actions]
a2_traj = np.zeros([T, 2, 2])   # [n_iter, n_state, n_actions]
incentive_traj = np.zeros([ T, 2, 2 ])   # [n_iter, n_state, n_actions ]
V_vec_traj = np.zeros([T, 2])   # [n_iter, n_state]
update_term_traj = np.zeros( [T, 2, 2] )    # [n_iter, n_state, n_actions]

for t in trange(T):
    if t >= 1000:
        print("Hey")
    incentive = np.zeros([2,2,2]) # [n_state, n_agent, n_agent_actions]
    for s in [0,1]: # Note that incentive only directly impact agents' cooperate action
        incentive[s, 0, 0] = ID_policy[s, 0]    # agent 1
        incentive[s, 1, 0] = ID_policy[s, 1]    # agent 2

    # Update agents strategies and incentive policy
    for s in [0,1]:
        a1[s], a2[s] = update_strategies(s, a1[s], a2[s], incentive[s][0], incentive[s][1], lamda, lr_a)

    # Obtain \nabal_a Q
    nabla_a_Q = get_nabla_a_Q(a1, a2, gamma)    
    
    # Calculate the sensitivity matrix
    sensitivity_mat = np.zeros([2, 2, 4])   # [n_state, n_theta, n_action]
    for s in [0,1]:
        sensitivity_mat[s] = get_sensetivity_mat(s, a1[s], a2[s], lamda)

    # Calculate update term of incentive
    for s in [0,1]:
        update_term = nabla_a_Q[s] @ sensitivity_mat[s].T - delta * ID_policy[s]    # Original update rule plus an regularization term
        ID_policy[s][0] = np.clip(ID_policy[s][0]+lr_i * update_term[0], theta_min, theta_max)
        ID_policy[s][1] = np.clip(ID_policy[s][1]+lr_i * update_term[1], theta_min, theta_max)
        update_term_traj[t][s] = update_term

    # Record trajectory
    for s in [0,1]:
        a1_traj[t, s] = a1[s]
        a2_traj[t, s] = a2[s]
        incentive_traj[t, s] = ID_policy[s]




colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# Plot agents' strategies trajectory
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10,8])
# fig.suptitle("Agents' strategies trajectory")
# dim_actions = incentive_traj[-1].shape[1]

# ax1 = axs[0][0]
# ax1.plot(np.arange(T), np.array(a1_traj)[:, 0, 0], label="coop", color='r')
# ax1.plot(np.arange(T), np.array(a1_traj)[:, 0, 1], label="betray", color='b')
# ax1.set_title("Agent 1 at COOP")
# ax1.legend()

# ax2 = axs[0][1]
# ax2.plot(np.arange(T), np.array(a1_traj)[:, 1, 0], label="coop", color='r')
# ax2.plot(np.arange(T), np.array(a1_traj)[:, 1, 1], label="betray", color='b')
# ax2.set_title("Agent 1 at BETRAY")
# ax2.legend()

# ax3 = axs[1][0]
# ax3.plot(np.arange(T), np.array(a2_traj)[:, 0, 0], label="coop", color='r')
# ax3.plot(np.arange(T), np.array(a2_traj)[:, 0, 1], label="betray", color='b')
# ax3.set_title("Agent 2 at COOP")
# ax3.legend()

# ax4 = axs[1][1]
# ax4.plot(np.arange(T), np.array(a2_traj)[:, 1, 0], label="coop", color='r')
# ax4.plot(np.arange(T), np.array(a2_traj)[:, 1, 1], label="betray", color='b')
# ax4.set_title("Agent 2 at BETRAY")
# ax4.legend()





# Plot incentive trajectory
# fig = plt.figure(figsize=[8,8])
# plt.title("Incentive trajectory")
# for s in [0,1]:
#     for a in [0,1]:
#         plt.plot(np.arange(T), incentive_traj[:, s, a], label="s{} a{}".format(s, a), color=colors[2*s+a])
# plt.legend()


# Calculate the gradient \nabla_\theta V_\rho(a_*(\theta))
gradient_sum = np.zeros([2,2])  # [n_state, n_theta]
grad_traj = np.zeros([T, 2, 2]) # [n_iter, n_state, n_theta]
sum_traj = np.zeros([T])  # [n_iter, n_state, n_theta]
nash_traj = np.zeros([T, 2, 2, 2])   # [n_iter, n_agent, n_state, n_agent_actions]
for t in range(T):
    incentive = incentive_traj[t]

    nash_a1, nash_a2 = get_nash(incentive)
    nash_traj[t, 0] = nash_a1
    nash_traj[t, 1] = nash_a2
    # Obtain \nabal_a Q
    nabla_a_Q = get_nabla_a_Q(nash_a1, nash_a2, gamma)    
    
    # Calculate the sensitivity matrix
    sensitivity_mat = np.zeros([2, 2, 4])   # [n_state, n_theta, n_action]
    for s in [0,1]:
        sensitivity_mat[s] = get_sensetivity_mat(s, nash_a1[s], nash_a2[s], lamda)

    # calculate the gradient
    cur_gradient = np.zeros([2,2])  # [n_state, n_theta]
    for s in [0,1]:
        cur_gradient[s] = nabla_a_Q[s] @ sensitivity_mat[s].T - delta * ID_policy[s]    # Original update rule plus an regularization term
    # cur_gradient *= 100
    grad_traj[t] = cur_gradient
    
    gradient_sum = gradient_sum * (t)/(t+1) + cur_gradient/(t+1)
    # sum_traj[t] = np.linalg.norm(gradient_sum)
    sum_traj[t] = np.linalg.norm(cur_gradient)
plt.figure(figsize=[8,8])
# plt.subplot(131)
plt.title("running average of gradient")
plt.plot(np.arange(T), sum_traj)
# plt.plot(np.arange(T), grad_traj[:,0,0], label="s0 a0")
# plt.plot(np.arange(T), grad_traj[:,0,1], label="s0 a1")
# plt.plot(np.arange(T), grad_traj[:,1,0], label="s1 a0")
# plt.plot(np.arange(T), grad_traj[:,1,1], label="s1 a1")
plt.xlabel("iteration")
plt.ylabel("gradient")
plt.legend()

# plt.subplot(132)
# plt.title("nash_a1")
# plt.plot(np.arange(T), nash_traj[:, 0, 0, 0], label="s0 a0")
# plt.plot(np.arange(T), nash_traj[:, 0, 0, 1], label="s0 a1")
# plt.plot(np.arange(T), nash_traj[:, 0, 1, 0], label="s1 a0")
# plt.plot(np.arange(T), nash_traj[:, 0, 1, 1], label="s1 a1")
# plt.legend()

# plt.subplot(133)
# plt.title("nash_a2")
# plt.plot(np.arange(T), nash_traj[:, 1, 0, 0], label="s0 a0")
# plt.plot(np.arange(T), nash_traj[:, 1, 0, 1], label="s0 a1")
# plt.plot(np.arange(T), nash_traj[:, 1, 1, 0], label="s1 a0")
# plt.plot(np.arange(T), nash_traj[:, 1, 1, 1], label="s1 a1")
# plt.legend()
# plt.show()



# ax7 = axs[1][2]
# ax7.plot(np.arange(T), V_vec_traj[:, 0], label="COOP", color=colors[0])
# ax7.plot(np.arange(T), V_vec_traj[:, 1], label="BETRAY", color=colors[1])
# ax7.set_title("V value traj")
# ax7.legend()

# ax8 = axs[1][3]
# for s in [0,1]:
#     ax8.plot(np.arange(T), update_term_traj[:, s, 0], color=colors[2*s], label="s{} a1".format(s))
#     ax8.plot(np.arange(T), update_term_traj[:, s, 1], color=colors[2*s+1], label="s{} a2".format(s))
# ax8.legend()

plt.show()
