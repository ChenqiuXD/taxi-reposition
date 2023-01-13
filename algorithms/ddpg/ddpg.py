# reference: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py 
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.ddpg.model import (Actor, Critic)
from algorithms.ddpg.random_process import OrnsteinUhlenbeckProcess
from algorithms.base_agent import BaseAgent

criterion = nn.MSELoss()

class DDPG(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        if args.seed > 0:
            self.seed(args.seed)

        self.device = args.device
        self.dim_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.dim_actions= env_config["num_nodes"]        # Action vector's dimension
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        self.actor = Actor(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.actor_target = Actor(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.dim_states, self.dim_actions).to(self.device)
        self.critic_target = Critic(self.dim_states, self.dim_actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=10*self.lr)
        
        #Create replay buffer
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer_ptr = 0
        self.buffer = np.zeros([self.buffer_size, 2*self.dim_states+self.dim_actions+2])  # each transition is [s,a,r,s',d] where d represents done
        self.random_process = OrnsteinUhlenbeckProcess(size=self.dim_actions, theta=0.15, mu=0., sigma=0.01)
        self.buffer_high = 0    # Used to mark the appended buffer (avoid training with un-appended sample indexes)

        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.gamma

        # Randomness parameters
        self.epsilon_max = 0.20
        self.epsilon_min = 0.10
        # TODO: currently the annealing parameters are fixed, could this be integrated into run_this.py? Or is it necessary?
        self.depsilon = (self.epsilon_max - self.epsilon_min) / 10000.
        self.epsilon = self.epsilon_max
        self.is_training = True

    def choose_action(self, s, is_random=False):
        if is_random:
            actions = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
        else:
            state = torch.FloatTensor(self.s2obs(s)).to(self.device)
            actions = self.actor(state).cpu().data.numpy()
            # actions += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
            actions = ( actions + np.random.normal(0, self.epsilon, self.dim_actions) ).clip(self.min_bonus, self.max_bonus)
            self.epsilon = np.maximum(self.epsilon-self.depsilon, self.epsilon_min) 
        
        return actions

    def append_transition(self, s, a, r, d, s_, info):
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate([self.s2obs(s), a, [r], self.s2obs(s_), [d]])
        self.buffer_ptr += 1
        self.buffer_high = np.minimum( self.buffer_high+1, self.buffer_size )
        # Buffer_high would not exceeds so that sample_index in sample_data() would reamin in self.buffer

    def learn(self):
        self.prep_train()
        # Sample batch
        if self.buffer_high >= self.batch_size:
            sample_index = np.random.choice( self.buffer_high, self.batch_size, replace=False ) # Cannot sample replicate samples. 
            batch_memory = self.buffer[sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.dim_states]).to(self.device)
            batch_actions = torch.FloatTensor(batch_memory[:, self.dim_states:self.dim_states+self.dim_actions]).to(self.device)
            batch_rewards = torch.FloatTensor(batch_memory[:, self.dim_states+self.dim_actions+1:self.dim_states+self.dim_actions+2]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_states-1:-1]).to(self.device)
            batch_done = torch.FloatTensor(1 - batch_memory[:, -1]).to(self.device).view([-1,1])
        else:   # Continue append transition, stop learn()
            print("Transition number is lesser than batch_size, continue training. ")
            return

        target_Q = self.critic_target( batch_next_state, self.actor_target(batch_next_state) )
        target_Q = batch_rewards + (batch_done * self.discount * target_Q).detach()

        # Curr q values
        curr_Q = self.critic(batch_state, batch_actions)
        
        # Update critic
        critic_loss = F.mse_loss(curr_Q, target_Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        actor_loss = -self.critic( batch_state, self.actor(batch_state) ).mean()

        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        return np.concatenate([s['idle_drivers']/100.0, s['demands']/100.0])

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

    def reset(self):
        self.random_process.reset_states()

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
