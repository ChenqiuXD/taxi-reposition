import os
from copy import deepcopy
import gym
import numpy as np
import torch
from tqdm import tqdm

from utils_for_dqn import *

class Config:  # for off-policy
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = True  # whether off-policy or on-policy of DRL algorithm

        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(64)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(512)  # collect horizon_len step while exploring, then update network
        self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
        self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(1e2)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_target'
        os.makedirs(self.cwd, exist_ok=True)


class AgentBase():
    def __init__(self, net_dims, state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau

        self.last_state = None  # save the last state of the trajectory for training. `last_state.shape == (state_dim)`
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

class AgentDQN(AgentBase):
    def __init__(self, net_dims, state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        self.act_target.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def explore_env(self, env, horizon_len: int, if_random: bool = False):
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, 1), dtype=torch.int32).to(self.device)
        rewards = torch.ones(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state

        get_action = self.act_target.get_action # !!!Change here, use target network to get action
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            if if_random:
                action = torch.randint(self.action_dim, size=(1,))[0]
            else:
                action = get_action(state.unsqueeze(0))[0, 0]

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, tuncated, _ = env.step(ary_action)
            if done:
                ary_state, info = env.reset()

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer):
        obj_critics = 0.0
        q_values = 0.0

        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for i in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            obj_critics += obj_critic.item()
            q_values += q_value.item()
        return obj_critics / update_times, q_values / update_times

    def get_obj_critic(self, buffer, batch_size: int):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)
            next_q = self.cri_target(next_state).max(dim=1, keepdim=True)[0]
            q_label = reward + undone * self.gamma * next_q
        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value.mean()

def train_agent(args: Config, max_episodes: int, repeat_times: int):
    score_traj = []

    for repeat in tqdm(range(repeat_times)):
        args.init_before_training()
        torch.set_grad_enabled(False)

        env = build_env(args.env_class, args.env_args)
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.last_state, info = env.reset()

        buffer = ReplayBuffer(
            gpu_id=args.gpu_id,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
        buffer.update(buffer_items)  # warm up for ReplayBuffer

        evaluator = Evaluator(
            eval_env=build_env(args.env_class, args.env_args),
            eval_per_step=args.eval_per_step,
            eval_times=args.eval_times,
            cwd=args.cwd,
        )

        scores = []
        for episode in tqdm(range(max_episodes), leave=False):  # start training
            buffer_items = agent.explore_env(env, args.horizon_len)
            buffer.update(buffer_items)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            # Evaluate and save cumulative reward
            cumulative_reward = evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
            if cumulative_reward != None:
                scores.append(cumulative_reward)
        score_traj.append(scores)

    np.save(f'{args.cwd}/scores.npy', score_traj)

if __name__ == '__main__':
    env_args = {
        'env_name': 'CartPole-v0',  # A pole is attached by an un-actuated joint to a cart.
        'state_dim': 4,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 2,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)

    args = Config(agent_class=AgentDQN, env_class=gym.make, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    max_episodes = 105
    repeat_times = 2
    train_agent(args, max_episodes, repeat_times)