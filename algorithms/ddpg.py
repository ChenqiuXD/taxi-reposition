import os
import pickle
import copy
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.base_agent import BaseAgent

from algorithms.utils.replay_buffer import ReplayBuffer

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
        x = (self.max_action-self.min_action) * (torch.tanh(self.l3(x))+1) /2 + self.min_action
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        else:
            self.use_cuda = False
        if args.seed > 0:
            self.seed(args.seed)

        self.device = args.device
        self.dim_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.dim_actions= env_config["num_nodes"]        # Action vector's dimension
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        # Construct network
        self.actor = Actor(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.critic = Critic(self.dim_states, self.dim_actions).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Hyper-parameters
        self.buffer = ReplayBuffer(size=args.buffer_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.discount = args.gamma

        # Utils
        self.time_counter = 0
        self.actor_update_counter = 0
        self.critic_update_counter = 0
        self.max_iter_num = args.num_env_steps
        self.is_training = True

        self.exploration_noise = args.epsilon
        self.normalization_factor = args.normalization_factor

    def choose_action(self, obs, is_random=False):
        if is_random:
            return np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            obs = self.s2obs(obs)
            action = self.actor(obs).cpu().data.numpy().flatten()
            action = (action+np.random.normal(0, self.exploration_noise, self.dim_action)).clip(self.min_bonus, self.max_bonus)
            return action
    
    def learn(self):
        self.time_counter += 1
        if self.time_counter<=2*self.batch_size:
            return 
    
        self.prep_train()
        data_info, data_batch = self.buffer.sample(self.batch_size)
        state = self.s2obs([data_batch[i][0] for i in range(len(data_batch))]).to(self.device)
        action = torch.FloatTensor(np.vstack([data_batch[i][1] for i in range(len(data_batch))])).to(self.device)
        action = (action-self.min_bonus)/(self.max_bonus-self.min_bonus)*2-1    # Normalize the action to [-1, 1]
        reward = torch.FloatTensor([data_batch[i][2] for i in range(len(data_batch))]).to(self.device)
        next_state = self.s2obs([data_batch[i][3] for i in range(len(data_batch))]).to(self.device)
        done = torch.FloatTensor([data_batch[i][4] for i in range(len(data_batch))]).to(self.device)

        # Critic update
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward.view([-1,1]) + ((1-done).view([-1,1]) * self.gamma * target_Q)
        cur_Q = self.critic(state, action)
        critic_loss = F.mse_loss(cur_Q, target_Q)
        self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.critic_update_counter)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.actor_update_counter)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.critic_update_counter += 1
        self.actor_update_counter += 1
        
        if self.critic_update_counter % (self.max_iter_num/10) == 0:
            self.exploration_noise *= 0.8

    def append_transition(self, s, a, r, d, s_, info):
        # Append transition
        self.buffer.add(s, a, r, s_, d)

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) or handles a list of such states """
        if isinstance(s, dict):
            obs = np.concatenate([s['idle_drivers'], s['demands']])
            return torch.FloatTensor(obs)/self.normalization_factor
        elif isinstance(s, list) and all(isinstance(state, dict) for state in s):
            obs_list = np.zeros([len(s), self.dim_states])
            for i, state in enumerate(s):
                obs_list[i] = np.concatenate([state['idle_drivers'], state['demands']])
            return torch.FloatTensor(obs_list)/self.normalization_factor
        else:
            raise ValueError("Input must be a dict or a list of dicts.")
    
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
    
    def seed(self, s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)

    def load_model(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
                os.path.join(output, "actor.pkl")
        )
        torch.save(
            self.critic.state_dict(),
                os.path.join(output, "critic.pkl")
        )