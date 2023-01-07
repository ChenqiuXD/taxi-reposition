
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from algorithms.ddpg.model import (Actor, Critic)
# from memory import SequentialMemory
from algorithms.ddpg.random_process import OrnsteinUhlenbeckProcess
from algorithms.ddpg.util import *
from algorithms.base_agent import BaseAgent

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(BaseAgent):
    def __init__(self, args, env_config):
        self.args = args
        if str(args.device) == 'cuda':
            self.use_cuda = True
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = 2 * env_config["num_nodes"]    # State vector's dimension
        self.nb_actions= env_config["num_nodes"]        # Action vector's dimension
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':400, 
            'hidden2':400, 
            'init_w':0.03
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.lr)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.lr*3)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        # self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer_ptr = 0
        self.buffer = np.zeros([self.buffer_size, 2*self.nb_states+self.nb_actions+2])  # each transition is [s,a,r,s',d] where d represents done
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=0.15, mu=0., sigma=0.01)
        self.buffer_high = 0

        # Hyper-parameters
        self.tau = args.tau
        self.discount = args.gamma
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.is_training = True

        # 
        if self.use_cuda: self.cuda()

    def append_transition(self, s, a, r, d, s_, info):
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0
        self.buffer[self.buffer_ptr] = np.concatenate([self.s2obs(s), a, [r], self.s2obs(s_), [d]])
        self.buffer_ptr += 1
        self.buffer_high = np.minimum( self.buffer_high+1, self.buffer_size-1 )

    def sample_data(self):
        sample_index = np.random.choice( self.buffer_high, self.batch_size )
        batch_memory = self.buffer[sample_index, :]
        batch_state = batch_memory[:, :self.nb_states]
        batch_actions = batch_memory[:, self.nb_states:self.nb_states+self.nb_actions]
        batch_rewards = batch_memory[:, self.nb_states+self.nb_actions+1:self.nb_states+self.nb_actions+2]
        batch_next_state = batch_memory[:, -self.nb_states-1:-1]
        batch_done = 1 - batch_memory[:, -1]
        return batch_state, batch_actions, batch_rewards, batch_next_state, batch_done

    def learn(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.sample_data()

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def choose_action(self, s, is_random=False):
        obs = self.s2obs(s)
        if is_random:
            action = np.random.uniform(-1., 1., self.nb_actions)
        else:
            # action = to_numpy(
            #     self.actor(to_tensor(obs))
            # ).squeeze(0)
            action = to_numpy( self.actor(to_tensor(obs)) )
            action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
            action = np.clip(action, -1., 1.)

        self.epsilon -= self.depsilon
        
        return action

    def s2obs(self, s):
        """ This function converts state(dict) to observation(ndarray) """
        return np.concatenate([s['idle_drivers'], s['demands']])

    def reset(self):
        self.random_process.reset_states()

    def load_weights(self, output):
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
