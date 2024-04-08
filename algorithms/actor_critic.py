# reference: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py 
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.base_agent import BaseAgent

criterion = nn.MSELoss()
NORMAALIZATION_FACTOR = 100.0

class A2CAgent(BaseAgent):
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

        self.action_set = np.array([0,1,2])
        self.num_actions = np.power(self.action_set.shape[0], self.dim_actions)    # Each node has [0,1,2], self.dim_actions=5 nodes
        self.actor = Actor(self.dim_states, self.num_actions).to(self.device)
        self.actor_target = Actor(self.dim_states, self.num_actions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.dim_states).to(self.device)
        self.critic_target = Critic(self.dim_states).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=10*self.lr)
        
        #Create replay buffer
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer_ptr = 0
        self.buffer = np.zeros([self.buffer_size, 2*self.dim_states+self.dim_actions+2])  # each transition is [s,a,r,s',d] where d represents done
        self.buffer_high = 0    # Used to mark the appended buffer (avoid training with un-appended sample indexes)

        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.gamma

    def choose_action(self, s, is_random=False):
        if is_random:
            actions = np.random.choice(a=np.arange(self.num_actions), size=self.dim_actions)
        else:
            state = torch.FloatTensor(self.s2obs(s)).to(self.device)
            actions_prob = self.actor(state).cpu().detach().numpy()
            actions = np.random.choice(a=np.arange(self.num_actions), p=actions_prob, size=1)[0]
            actions = self.convert_actions(actions)
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
            batch_actions = batch_memory[:, self.dim_states:self.dim_states+self.dim_actions]
            batch_rewards = torch.FloatTensor(batch_memory[:, self.dim_states+self.dim_actions:self.dim_states+self.dim_actions+1]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.dim_states-1:-1]).to(self.device)
            batch_done = torch.FloatTensor(1 - batch_memory[:, -1]).to(self.device).view([-1,1])
        else:   # Continue append transition, stop learn()
            print("Transition number is lesser than batch_size, continue training. ")
            return

        # Target q values
        target_V = self.critic_target( batch_next_state )
        target_V = batch_rewards + (batch_done * self.discount * target_V).detach()

        # Curr q values
        curr_V = self.critic(batch_state)
        
        # Update critic
        td_error = target_V - curr_V
        critic_loss = td_error.pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        action_index = torch.LongTensor( self.convert_actions(batch_actions) ).to(self.device).view([-1,1])
        action_prob = self.actor(batch_state).gather(1, action_index)
        actor_loss = - (torch.log(action_prob) * td_error.detach()).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Record the loss trajectory
        critic_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.critic.parameters()])
        self.writer.add_scalar("critic_grad_max", np.max(critic_grad), self.train_steps)
        actor_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.actor.parameters()])
        self.writer.add_scalar("actor_grad_max", np.max(actor_grad), self.train_steps)
        self.writer.add_scalar("critic_loss", critic_loss.item(), self.train_steps)
        self.writer.add_scalar("actor_loss", actor_loss.item(), self.train_steps)
        self.train_steps += 1
    
        # Soft update the target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def convert_actions(self, num):
        """ This function converts the actions to action index """
        # TODO: Here the conversion is to [0,1,2,3], please change it to self.actions_list
        num_bonus = self.action_set.shape[0]
        if type(num) == np.ndarray:   # Convert a list action to a number
            actions = np.zeros(num.shape[0])
            for idx in range(self.dim_action):
                actions += num_bonus**idx * num[:, idx]
        else:   # Convert number to a list of actions
            tmp = num
            actions = np.zeros(self.num_nodes)
            idx = 0
            while tmp != 0:
                actions[idx] = tmp%num_bonus
                tmp = int(tmp/num_bonus)
                idx += 1
        return actions

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        return np.concatenate([s['idle_drivers']/NORMAALIZATION_FACTOR, s['demands']/NORMAALIZATION_FACTOR])

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
    def __init__(self, state_dim, num_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 2096)
        self.l2 = nn.Linear(2096, 1024)
        self.l3 = nn.Linear(1024, num_action)

    def forward(self, x, softmax_dim=0):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob


class Critic(nn.Module):
    def __init__(self, state_dim):
        """ V-value function """
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 2096)
        self.l2 = nn.Linear(2096, 1024)
        self.l3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x