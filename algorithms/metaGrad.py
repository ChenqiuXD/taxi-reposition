# reference: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py 
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from environment.utils import get_adj_mat

from algorithms.base_agent import BaseAgent

criterion = nn.MSELoss()
NORMALIZATION_FACTOR = 100.0

class metaAgent(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        if args.seed > 0:
            self.seed(args.seed)

        self.device = args.device
        self.episode_length = args.episode_length
        self.dim_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.dim_actions = env_config["num_nodes"]        # Action vector's dimension
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        # adjangency related parameters
        self.adj_mat = get_adj_mat(self.edge_index)
        self.neighbour_list = [np.where(self.adj_mat[i]==1)[0] for i in range(self.num_nodes)]
        dim_policies = np.sum([len(neighbour) for neighbour in self.neighbour_list])

        # actor and critic neural network
        self.actor = Actor(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.actor_target = Actor(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.dim_states, dim_policies).to(self.device)
        self.critic_target = Critic(self.dim_states, dim_policies).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=10*self.lr)

        #Create replay buffer
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer_ptr = 0
        self.buffer = np.zeros([self.buffer_size, 2*self.dim_states+self.dim_actions+2])  # each transition is [s,a,r,s',d] where d represents don
        self.buffer_agents_actions = np.zeros([self.buffer_size, dim_policies])    # Record history agents' actions. [buffer_size, dim_agents_policies]
        self.curr_agents_policies = np.zeros([self.episode_length, dim_policies])   # Record current iteration agents' policies. [episode_length, dim_agents_policies]
        self.buffer_time_steps = np.zeros([self.buffer_size])
        self.buffer_high = 0    # Used to mark the appended buffer (avoid training with un-appended sample indexes)
        
        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.gamma

        # Randomness parameters
        self.epsilon_max = args.max_epsilon
        self.epsilon_min = args.min_epsilon
        self.depsilon = (self.epsilon_max - self.epsilon_min) / args.decre_epsilon_episodes
        self.epsilon = self.epsilon_max
        self.is_training = True

        # sensitivity related constant
        n_neighbour = np.array([neighbour.shape[0] for neighbour in self.neighbour_list])
        self.A = np.zeros([self.num_nodes, dim_policies])    # constraint matrix: only constraint that policies sums are 1
        cnt = 0
        for i in range(self.num_nodes):
            self.A[i, cnt:cnt+n_neighbour[i]] = np.ones(n_neighbour[i])
            cnt += n_neighbour[i]
        self.nabla_y_F = np.concatenate([np.eye(self.num_nodes)[self.neighbour_list[i]].T 
                                        for i in range(self.num_nodes)], axis=1)
                                                # calculate the S matrix
        n_node = env_config['num_nodes']
        self.S = []
        for i in range(n_node):
            tmp = np.zeros([n_node, n_node])
            for j in range(n_node):
                if j in self.neighbour_list[i]:
                    tmp[j, j] += 1
            self.S.append(tmp)

    def choose_action(self, s, is_random=False):
        if is_random:
            actions = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            state = torch.FloatTensor(self.s2obs(s)).to(self.device)
            actions = self.actor(state).cpu().detach().numpy()
            actions = ( actions + np.random.normal(0, self.epsilon, self.dim_actions) ).clip(self.min_bonus, self.max_bonus)
            self.epsilon = np.maximum(self.epsilon-self.depsilon, self.epsilon_min) 
        
        return actions

    def append_transition(self, s, a, r, d, s_, info):
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate([self.s2obs(s), a, [r], self.s2obs(s_), [d]])
        self.buffer_high = np.minimum( self.buffer_high+1, self.buffer_size )
        # Buffer_high would not exceeds so that sample_index in sample_data() would reamin in self.buffer

        # Record agents distribution policies
        agent_actions = np.concatenate( [ arr[self.neighbour_list[idx]] / np.sum(arr)
                                             for idx, arr in enumerate(info[0]) ] ) # [dim_policies], record the drivers' policies by vector
        self.buffer_agents_actions[self.buffer_ptr] = agent_actions   # Record history agents' actions. [buffer_size, dim_agents_policies]
        self.curr_agents_policies[s['time_step']] = agent_actions  # Record current agents' policies. [episode_length, dim_agents_policies]
        self.buffer_time_steps[self.buffer_ptr] = s['time_step']
        self.buffer_ptr += 1

    def learn(self):
        self.prep_train()
        # Sample batch
        if self.buffer_high >= self.batch_size:
            sample_index = np.random.choice( self.buffer_high, self.batch_size, replace=False ) # Cannot sample replicate samples. 
            batch_memory = self.buffer[sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.dim_states]).to(self.device)
            # batch_actions = torch.FloatTensor(batch_memory[:, self.dim_states:self.dim_states+self.dim_actions]).to(self.device)
            batch_rewards = torch.FloatTensor(batch_memory[:, self.dim_states+self.dim_actions:self.dim_states+self.dim_actions+1]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_states-1:-1]).to(self.device)
            batch_done_np = 1 - batch_memory[:, -1]
            batch_done = torch.FloatTensor(batch_done_np).to(self.device).view([-1,1])

            batch_agents_policies = self.buffer_agents_actions[sample_index]
            batch_idle_drivers = batch_memory[:, :self.num_nodes]
            batch_demands = batch_memory[:, self.num_nodes: self.num_nodes * 2]
        else:   # Continue append transition, stop learn()
            print("Transition number is lesser than batch_size, continue training. ")
            return

        batch_next_actions = np.array([ self.curr_agents_policies[\
                                    self.buffer_time_steps[int( (i_sample+1)*batch_done_np[idx] )].astype(int) ]
                                 for idx, i_sample in enumerate(sample_index) ])
        target_Q = self.critic_target( batch_next_state, torch.FloatTensor(batch_next_actions).to(self.device) )
        target_Q = batch_rewards + (batch_done * self.discount * target_Q).detach()

        # Curr q values
        curr_Q = self.critic(batch_state, torch.FloatTensor(batch_agents_policies).to(self.device))
        
        # Update critic
        critic_loss = F.mse_loss(curr_Q, target_Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Record maximum of critic gradient
        critic_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.critic.parameters()])
        self.writer.add_scalar("critic_grad_max", np.max(critic_grad), self.train_steps)
        self.writer.add_scalar("critic_loss", critic_loss.item(), self.train_steps)

        # Update actor network
        for idx, state in enumerate(batch_state):
            # Try for separate update, first calculate \nabla_\theta \mu(s)
            actor_criterion = self.actor(state)
            grads = []
            for k in range(self.num_nodes): # backward() could only be performed on scalar, thus we use output of each node to calculate backward()
                self.actor_optim.zero_grad()
                actor_criterion[k].backward(retain_graph=True)
                grads.append(torch.cat([ p.grad.flatten().clone().detach() for p in self.actor.parameters() ], dim=0))
            actor_grads = torch.vstack(grads)

            # Calculate the sensitivity matrix
            agents_policy = batch_agents_policies[idx]
            L = self.get_L(agents_policy, batch_idle_drivers[idx]*NORMALIZATION_FACTOR, batch_demands[idx]*NORMALIZATION_FACTOR)
            M = L-L@self.A.T@ np.linalg.inv(self.A@L@self.A.T) @ self.A@L
            nabla_y_x = (-M @ self.nabla_y_F.T).T

            # then calculate the \nabla_a Q(s,a)
            # TODO: test whether zero_grad is necessary
            # self.actor_optim.zero_grad()
            # self.critic_optim.zero_grad()
            batch_actions_mu = torch.FloatTensor(agents_policy).to(self.device).requires_grad_(True)
            batch_actions_mu.retain_grad()
            critic_criteriion = - self.critic(state, batch_actions_mu)
            critic_criteriion.backward()

            # Calculate the comprehensive gradient
            if 'grad' in locals().keys():
                grad += actor_grads.T @ torch.FloatTensor(nabla_y_x).to(self.device) @ batch_actions_mu.grad / self.batch_size
            else:
                grad = actor_grads.T @ torch.FloatTensor(nabla_y_x).to(self.device) @ batch_actions_mu.grad / self.batch_size

        # Reshape grads to actor.parameters() shapes
        shapes = [x.shape for x in self.actor.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]
        grad_split = grad.split(shapes_prod)
        cnt = 0
        for n, p in self.actor.named_parameters():  # Assign gradient to actor network. 
            p.grad = grad_split[cnt].view(p.grad.shape)
            cnt += 1
        
        # actor update
        self.actor_optim.step()

        # Record actor gradient
        actor_grad = torch.concat(grad_split)
        self.writer.add_scalar("actor_grad_max", torch.max(actor_grad).item(), self.train_steps)
        self.train_steps += 1

        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_L(self, policies, idle_drivers, demands):
        """ This function calculate the L matrix """
        n_node = idle_drivers.shape[0]
        n_neighbour = np.array([neighbour.shape[0] for neighbour in self.neighbour_list])
        dim_policies = np.sum(n_neighbour)

        L = np.zeros([dim_policies, dim_policies])
        cnt_col = 0
        for i in range(n_node):
            cnt_row = 0
            for j in range(n_node):
                sub_mat = (1+(i==j)) * idle_drivers[i] * self.S[j][self.neighbour_list[j]][:, self.neighbour_list[i]]
                L[cnt_row:cnt_row+n_neighbour[j], cnt_col:cnt_col+n_neighbour[i]] = sub_mat
                cnt_row += n_neighbour[j]
            for idx, neighbour_node in enumerate(self.neighbour_list[i]):
                L[:, cnt_col+idx:cnt_col+idx+1] /= demands[neighbour_node]            
            cnt_col += n_neighbour[i]

        # Entropy would change the diagonal of L
        L += np.diag(1/policies)
        return -L

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        return np.concatenate([s['idle_drivers']/NORMALIZATION_FACTOR, s['demands']/NORMALIZATION_FACTOR])

    def prep_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def load_model(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
                os.path.join(output, "actor.pkl")
        )
        torch.save(
            self.critic.state_dict(),
                os.path.join(output, "critic.pkl")
        )

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.min_action = min_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) 
        x = (x+1) * (self.max_action-self.min_action) / 2 + self.min_action # Normalize actions
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        try:
            x = F.relu(self.l1(torch.cat([x,u], 1)))
        except:
            x = F.relu(self.l1(torch.cat([x,u], 0)))
        # x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x