import os
import pickle
import copy
import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from algorithms.base_agent import BaseAgent

def compute_return_advantage(rewards, values, is_last_terminal, gamma, gae_lambda, last_value):
    """
    Computes returns and advantage based on generalized advantage estimation.
    """
    N = rewards.shape[0]
    advantages = np.zeros(
        (N, 1),
        dtype=np.float32
    )

    tmp = 0.0
    for k in reversed(range(N)):
        if k==N-1:
            next_non_terminal = 1 - is_last_terminal
            next_values = last_value
        else:
            next_non_terminal = 1
            next_values = values[k+1]

        delta = rewards[k] + gamma * next_non_terminal * next_values - values[k]
        tmp = delta + gamma * gae_lambda * next_non_terminal * tmp
        
        advantages[k] = tmp
    
    returns = advantages +  values

    return returns, advantages

class PPOBuffer:
    def __init__(self, obs_dim, action_dim, buffer_capacity, seed=None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_capacity = buffer_capacity

        self.obs = np.zeros(
            shape=(self.buffer_capacity, self.obs_dim),
            dtype=np.float32
        )
        self.action = np.zeros(
            shape=(self.buffer_capacity, self.action_dim),
            dtype=np.float32
        )
        self.reward = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.log_prob = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.returns = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.advantage = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.values = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )


        self.rng = np.random.default_rng(seed=seed)
        self.start_index, self.pointer = 0, 0

    def record(self, obs, action, reward, values, log_prob):
        self.obs[self.pointer] = obs
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.values[self.pointer] = values
        self.log_prob[self.pointer] = log_prob

        self.pointer += 1
        if self.pointer >= self.buffer_capacity:
            self.pointer = 0

    def process_trajectory(self, gamma, gae_lam, is_last_terminal, last_v):
        path_slice = slice(self.start_index, self.pointer)
        values_t = self.values[path_slice]

        self.returns[path_slice], self.advantage[path_slice] = compute_return_advantage(
            self.reward[path_slice],
            values_t,
            is_last_terminal,
            gamma,
            gae_lam,
            last_v
        )

        self.start_index = self.pointer

    def get_data(self):
        whole_slice = slice(0, self.pointer)
        return {
            'obs': self.obs[whole_slice],
            'action': self.action[whole_slice],
            'reward': self.reward[whole_slice],
            'values': self.values[whole_slice],
            'log_prob': self.log_prob[whole_slice],
            'return': self.returns[whole_slice],
            'advantage': self.advantage[whole_slice],
        }

    def get_mini_batch(self, batch_size):
        # assert batch_size <= self.pointer, "Batch size must be smaller than number of data."
        indices = np.arange(self.pointer)
        self.rng.shuffle(indices)
        

        split_indices = []
        point = batch_size
        while point < self.pointer:
            split_indices.append(point)
            point += batch_size

        temp_data = {
            'obs': np.split(self.obs[indices], split_indices),
            'action': np.split(self.action[indices], split_indices),
            'reward': np.split(self.reward[indices], split_indices),
            'values': np.split(self.values[indices], split_indices),
            'log_prob': np.split(self.log_prob[indices], split_indices),
            'return': np.split(self.returns[indices], split_indices),
            'advantage': np.split(self.advantage[indices], split_indices),
        }

        n = len(temp_data['obs'])
        data_out = []
        for k in range(n):
            data_out.append(
                {
                    'obs': temp_data['obs'][k],
                    'action': temp_data['action'][k],
                    'reward': temp_data['reward'][k],
                    'values': temp_data['values'][k],
                    'log_prob': temp_data['log_prob'][k],
                    'return': temp_data['return'][k],
                    'advantage': temp_data['advantage'][k],
                }
            )
        
        return data_out

    def clear(self):
        self.start_index, self.pointer = 0, 0

class PPOPolicy(nn.Module):
    def __init__(self, pi_network, v_network, learning_rate, obs_dim, action_dim, clip_range=0.2, value_coeff=0.5, initial_std=1.0, max_grad_norm=0.5) -> None:
        super().__init__()

        (
            self.pi_network,
            self.v_network,
            self.learning_rate,
            self.clip_range,
            self.value_coeff,
            self.obs_dim,
            self.action_dim,
            self.max_grad_norm,
        ) = (
            pi_network,
            v_network,
            learning_rate,
            clip_range,
            value_coeff,
            obs_dim,
            action_dim,
            max_grad_norm
        )

        # Gaussian policy will be used. So, log standard deviation is created as trainable variables
        self.log_std =  nn.Parameter(torch.ones(self.action_dim) * torch.log(torch.tensor(initial_std)), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        pi_out = self.pi_network(obs)

        # Add Normal distribution layer at the output of pi_network
        dist_out = Normal(pi_out, torch.exp(self.log_std))

        v_out = self.v_network(obs)

        return dist_out, v_out

    def get_action(self, obs):
        """
        Sample action based on current policy
        """
        obs_torch = torch.unsqueeze(obs, 0)
        dist, values = self.forward(obs_torch)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=1)

        return action[0].detach().numpy(), torch.squeeze(log_prob).detach().numpy(), torch.squeeze(values).detach().numpy()

    def get_values(self, obs):
        """
        Function  to return value of the state
        """
        obs_torch = torch.unsqueeze(obs, 0)

        _, values = self.forward(obs_torch)

        return torch.squeeze(values).detach().numpy()

    def evaluate_action(self, obs_batch, action_batch, training):
        """
        Evaluate taken action.
        """     
  
        obs_torch = obs_batch.clone().detach()
        action_torch = action_batch.clone().detach()
        dist, values = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return log_prob, values

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
        """
        Performs one step gradient update of policy and value network.
        """

        new_log_prob, values = self.evaluate_action(obs_batch, action_batch, training=True)

        ratio = torch.exp(new_log_prob-log_prob_batch)
        clipped_ratio = torch.clip(
            ratio,
            1-self.clip_range,
            1+self.clip_range,
        )

        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = self.value_coeff * torch.mean((values - return_batch)**2)
        total_loss = pi_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            pi_loss.detach(), 
            value_loss.detach(), 
            total_loss.detach(), 
            (torch.mean((ratio - 1) - torch.log(ratio))).detach(), 
            torch.exp(self.log_std).detach()
        )

class PI_Network(nn.Module):
    def __init__(self, obs_dim, action_dim, upper_bound, lower_bound) -> None:
        super().__init__()
        (
            self.lower_bound,
            self.upper_bound
        ) = (
            torch.tensor(lower_bound, dtype=torch.float32),
            torch.tensor(upper_bound, dtype=torch.float32)
        )
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        y = torch.tanh(self.fc1(obs))
        y = torch.tanh(self.fc2(y))
        action = self.fc3(y)

        action = (action + 1)*(self.upper_bound - self.lower_bound)/2+self.lower_bound

        return action

class V_Network(nn.Module):
    def __init__(self, obs_dim) -> None:
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        y = torch.tanh(self.fc1(obs))
        y = torch.tanh(self.fc2(y))
        values = self.fc3(y)

        return values

class PPO(BaseAgent):
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
        self.actor = PI_Network(self.dim_states, self.dim_actions, self.max_bonus, self.min_bonus).to(self.device)
        self.critic = V_Network(self.dim_states).to(self.device)
        self.policy = PPOPolicy(self.actor, self.critic, self.lr, self.dim_states, self.dim_actions)

        # Hyper-parameters
        self.buffer = PPOBuffer(self.dim_states, self.dim_actions, args.buffer_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau

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
            action = self.policy.get_action(obs)[0]
            action = np.clip(action, self.min_bonus, self.max_bonus)
            return action
    
    def learn(self):
        self.time_counter += 1
        if self.time_counter<=2*self.batch_size:
            return     
        self.prep_train()

        # Update for epochs
        for ep in range(10):    # for each epoch
            batch_data = self.buffer.get_mini_batch(self.batch_size)
            num_grads = len(batch_data)

            for k in range(num_grads):
                # Load sampled data
                data = batch_data[k]
                obs_batch = torch.tensor(data['obs'], dtype=torch.float32).to(self.device)
                action_batch = torch.tensor(data['action'], dtype=torch.float32).to(self.device)
                log_prob_batch = torch.tensor(data['log_prob'], dtype=torch.float32).to(self.device)
                return_batch = torch.tensor(data['return'], dtype=torch.float32).to(self.device)

                # Normalize advantage batch
                advantage_batch = data['advantage']
                advantage_batch = (advantage_batch - np.squeeze(advantage_batch.mean())) / (np.squeeze(advantage_batch.std()) + 1e-8)
                advantage_batch = torch.tensor(advantage_batch, dtype=torch.float32).to(self.device)
                
                # Update policy
                pi_loss, value_loss, total_loss, entropy, std = self.policy.update(obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch)
                
                self.actor_update_counter += 1
                self.critic_update_counter += 1

                self.writer.add_scalar('Loss/pi_loss', pi_loss, self.actor_update_counter)
                self.writer.add_scalar('Loss/value_loss', value_loss, self.critic_update_counter)
                self.writer.add_scalar('Loss/total_loss', total_loss, self.actor_update_counter)
                self.writer.add_scalar('Loss/entropy', entropy, self.actor_update_counter)
                self.writer.add_scalar('Loss/std', std.mean(), self.actor_update_counter)
            
            # self.buffer.clear() # TODO: why need clear the buffer? 
            


    def append_transition(self, s, a, r, d, s_, info):
        s = self.s2obs(s)
        _, log_prob, values = self.policy.get_action(s)
        self.buffer.record(s, a, r, values, log_prob)

        if d:   # TODO: check what this do?
            last_value = self.policy.get_values(self.s2obs(s_))
            self.buffer.process_trajectory(gamma=self.gamma, gae_lam=0.95, is_last_terminal=1, last_v=last_value)

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
        self.critic.train()
        
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