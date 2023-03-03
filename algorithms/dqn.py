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

class dqnAgent(BaseAgent):
    def __init__(self, args, env_config):
        super().__init__(args, env_config)
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        if args.seed > 0:
            self.use_cuda = False
            self.seed(args.seed)

        self.device = args.device
        self.dim_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.dim_actions= env_config["num_nodes"]        # Action vector's dimension
        self.max_bonus = args.max_bonus
        self.min_bonus = args.min_bonus

        self.bonus_choice = 4    # Possible selection of actions
        # self.actions_list = np.linspace(self.min_bonus, self.max_bonus, num=self.bonus_choice)
        # TODO: Here the self.actions_list is direclty assigned, please change it to adapt with respect to self.bonus_choice
        self.actions_list = np.array([0,1,2,3])
        self.num_actions = self.actions_list.shape[0]**self.num_nodes

        self.qnet = net(self.dim_states, self.num_actions).to(self.device)
        self.qnet_target = net(self.dim_states, self.num_actions).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optim = Adam(self.qnet.parameters(), lr=self.lr)
        
        #Create replay buffer
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer_ptr = 0
        self.buffer = np.zeros([self.buffer_size, 2*self.dim_states+self.dim_actions+2])  # each transition is [s,a,r,s',d] where d represents done
        self.buffer_high = 0    # Used to mark the appended buffer (avoid training with un-appended sample indexes)

        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.gamma

        # Randomness parameters
        self.epsilon = args.epsilon
        # TODO: Currently the self.epsilon would not decrease, please change it to be able to decrease. 
        self.is_training = True

    def choose_action(self, s, is_random=False):
        if is_random or np.random.uniform() < self.epsilon: # eps usually 0.1
            # actions = np.random.uniform(self.min_bonus, self.max_bonus, self.dim_actions)
            actions = np.random.choice(self.actions_list, size=[self.dim_action])
        else:
            state = torch.FloatTensor(self.s2obs(s)).to(self.device)
            values = self.qnet(state).cpu().detach().numpy()
            action_list = np.where( values == np.max(values) )[0]
            action_idx = np.random.choice(action_list)
            actions = self.convert_actions(action_idx)
        
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

        # Curr q values
        actions_encoded = torch.LongTensor( [ self.convert_actions(actions) for actions in batch_actions ] ).to(self.device).view([-1,1])
        curr_Q = self.qnet(batch_state).gather(1, actions_encoded)

        # Target q values
        q_next = self.qnet_target(batch_next_state).detach()
        q_target = batch_rewards + self.discount * batch_done * q_next.max(1)[0].view(self.batch_size, 1)
        
        # Update critic
        loss = F.mse_loss(curr_Q, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Record the loss trajectory
        critic_grad = np.concatenate([x.grad.cpu().numpy().reshape([-1]) for x in self.qnet.parameters()])
        self.writer.add_scalar("critic_grad_max", np.max(critic_grad), self.train_steps)
        self.writer.add_scalar("critic_loss", loss.item(), self.train_steps)
        self.train_steps += 1
    
        # Soft update the target network
        for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def convert_actions(self, num):
        """ This function converts the  """
        # TODO: Here the conversion is to [0,1,2,3], please change it to self.actions_list
        if type(num) == np.ndarray:   # Convert a list action to a number
            actions = 0
            for idx, element in enumerate(num):
                actions += self.bonus_choice**idx * element
        else:   # Convert number to a list of actions
            tmp = num
            actions = np.zeros(self.num_nodes)
            idx = 0
            while tmp != 0:
                actions[idx] = tmp%self.bonus_choice
                tmp = int(tmp/self.bonus_choice)
                idx += 1
        return actions

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        return np.concatenate([s['idle_drivers']/NORMAALIZATION_FACTOR, s['demands']/NORMAALIZATION_FACTOR])

    def prep_train(self):
        self.qnet.train()
        self.qnet_target.train()

    def eval(self):
        self.qnet.eval()
        self.qnet_target.eval()

    def load_model(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/qnet.pkl'.format(output))
        )

    def save_model(self,output):
        torch.save(
            self.qnet.state_dict(),
                os.path.join(output, "qnet.pkl")
        )

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)


class net(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(net, self).__init__()

        self.l1 = nn.Linear(state_dim, 2096)
        self.l2 = nn.Linear(2096 , 1600)
        self.l3 = nn.Linear(1600, num_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x