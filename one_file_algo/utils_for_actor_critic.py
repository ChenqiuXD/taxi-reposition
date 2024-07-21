import os
import sys
import time
from copy import deepcopy
from typing import Tuple, List, Optional

import numpy as np
import torch as th
import torch.nn as nn
import gym

from ac import Config

ARY = np.ndarray
TEN = th.Tensor

class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ActionDist = th.distributions.normal.Normal

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std


class Actor(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: TEN, action_std: float) -> TEN:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state).tanh()
        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


class ActorSAC(ActorBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.encoder_s = build_mlp(dims=[state_dim, *net_dims])  # encoder of state
        self.decoder_a_avg = build_mlp(dims=[net_dims[-1], action_dim])  # decoder of action mean
        self.decoder_a_std = build_mlp(dims=[net_dims[-1], action_dim])  # decoder of action log_std
        self.soft_plus = nn.Softplus()

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        return self.decoder_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: TEN, **_kwargs) -> TEN:  # for exploration
        state = self.state_norm(state)
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_avg = self.decoder_a_avg(state_tmp)
        action_std = self.decoder_a_std(state_tmp).clamp(-20, 2).exp()

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.tanh()  # action (re-parameterize)

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        state = self.state_norm(state)
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_log_std = self.decoder_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.decoder_a_avg(state_tmp)

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        logprob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))
        # dist = self.Normal(action_avg, action_std)
        # action = dist.sample()
        # logprob = dist.log_prob(action)

        '''fix logprob by adding the derivative of y=tanh(x)'''
        logprob -= (np.log(2.) - action - self.soft_plus(-2. * action)) * 2.  # better than below
        # logprob -= (1.000001 - action.tanh().pow(2)).log()
        return action.tanh(), logprob.sum(1, keepdim=True)


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std


class Critic(CriticBase):
    def __init__(self, net_dims, state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, 1])

    def forward(self, state: TEN, action: TEN) -> TEN:
        state = self.state_norm(state)
        value = self.net(th.cat((state, action), dim=1))
        return value  # Q value


class CriticTwin(CriticBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 8):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, num_ensembles])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        state = self.state_norm(state)
        values = self.net(th.cat((state, action), dim=1))
        return values  # Q values


class CriticEnsemble(CriticBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 8):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.encoder_sa = build_mlp(dims=[state_dim + action_dim, net_dims[0]])  # encoder of state and action
        self.decoder_qs = []
        for net_i in range(num_ensembles):
            decoder_q = build_mlp(dims=[*net_dims, 1])
            layer_init_with_orthogonal(decoder_q[-1], std=0.5)

            self.decoder_qs.append(decoder_q)
            setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        state = self.state_norm(state)
        tensor_sa = self.encoder_sa(th.cat((state, action), dim=1))
        values = th.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
        return values  # Q values


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


def build_mlp(dims: List[int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list
        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        action_dim = env.action_space.n if if_discrete else env.action_space.shape[0]
    else:
        env_name = env.env_name
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class, env_args: dict):
    if env_class.__module__ == 'gymnasium.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
        self.actions = th.empty((max_size, action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.empty((max_size, 1), dtype=th.float32, device=self.device)
        self.undones = th.empty((max_size, 1), dtype=th.float32, device=self.device)
        self.unmasks = th.empty((max_size, 1), dtype=th.float32, device=self.device)

    def update(self, items: Tuple[TEN, TEN, TEN, TEN, TEN]):
        states, actions, rewards, undones, unmasks = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
            self.unmasks[p0:p1], self.unmasks[0:p] = unmasks[:p2], unmasks[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
            self.unmasks[self.p:p] = unmasks
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return (
            self.states[ids],
            self.actions[ids],
            self.rewards[ids],
            self.undones[ids],
            self.unmasks[ids],
            self.states[ids + 1],
        )


class AgentBase:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args = Config()):
        self.net_dims: List[int] = net_dims
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.gamma: float = args.gamma
        self.batch_size: int = args.batch_size
        self.horizon_len: int = args.horizon_len
        self.repeat_times: float = args.repeat_times
        self.reward_scale: float = args.reward_scale
        self.learning_rate: float = args.learning_rate
        self.soft_update_tau: float = args.soft_update_tau
        self.state_update_tau: float = args.state_update_tau

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.last_state: Optional[ARY] = None  # state of the trajectory for training. `shape == (state_dim)`
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.act: Optional[ActorBase] = None
        self.cri: Optional[CriticBase] = None
        self.act_target = self.act
        self.cri_target = self.cri

        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = th.nn.SmoothL1Loss()

    def get_random_action(self) -> TEN:
        return th.rand(self.action_dim) * 2 - 1.0

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        self.horizon_len = horizon_len  # update horizon_len for update_net()

        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        ary_state = self.last_state
        for i in range(horizon_len):
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device)
            action = self.get_random_action() if if_random \
                else self.get_policy_action(state)

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            terminals[i] = terminal
            truncates[i] = truncate

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = th.logical_not(terminals).unsqueeze(1)
        unmasks = th.logical_not(truncates).unsqueeze(1)
        return states, actions, rewards, undones, unmasks

    def update_critic_net(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action = self.act(next_state)  # deterministic policy
            next_q = self.cri_target(next_state, next_action)

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def update_actor_net(self, state: TEN, update_t: int, if_skip: bool = False) -> Optional[TEN]:
        if if_skip:
            return None

        action_pg = self.act(state)  # action to policy gradient
        obj_actor = self.cri(state, action_pg).mean()
        return obj_actor

    def update_net(self, buffer, if_skip_actor: bool = False) -> Tuple[float, float]:
        states = buffer.states[-self.horizon_len:]
        state_update_tau = 1.0 if if_skip_actor else self.state_update_tau
        self.update_avg_std_for_state_norm(states=states, tau=state_update_tau)

        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, state = self.update_critic_net(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics.append(obj_critic.item())

            obj_actor = self.update_actor_net(state, update_t, if_skip_actor)
            if isinstance(obj_actor, TEN):
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
                obj_actors.append(obj_actor.item())
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg

    @staticmethod
    def optimizer_update(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        if target_net is current_net:
            return
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_avg_std_for_state_norm(self, states: TEN, tau: float):
        if tau == 0 or self.state_update_tau == 0:
            return
        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.cri.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std


class AgentDDPG(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.explore_noise_std = getattr(args, 'explore_noise', 0.05)  # set for `self.get_policy_action()`

        self.act = Actor(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = Critic(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]


class AgentTD3(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.update_freq = getattr(args, 'update_freq', 2)  # standard deviation of exploration noise
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.act = Actor(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticTwin(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

    def update_critic_net(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action = self.act.get_action(next_state, action_std=self.policy_noise_std)  # deterministic policy
            next_q = self.cri_target.get_q_values(next_state, next_action).min(dim=1, keepdim=True)[0]

            q_label = reward + undone * self.gamma * next_q

        q_values = self.cri.get_q_values(state, action) * unmask
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def update_actor_net(self, state: TEN, update_t: int = 0, if_skip: bool = False) -> Optional[TEN]:
        if if_skip:
            return None

        action_pg = self.act(state)  # action to policy gradient
        obj_actor = self.cri_target.get_q_values(state, action_pg).mean()
        return obj_actor


class AgentSAC(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks

        self.act = ActorSAC(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticEnsemble(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.alpha_log = th.tensor(-1, dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = -np.log(action_dim)

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0))[0]  # stochastic policy for exploration

    def update_critic_net(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1, keepdim=True)[0]
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        q_values = self.cri.get_q_values(state, action) * unmask
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def update_actor_net(self, state: TEN, update_t: int = 0, if_skip: bool = False) -> Optional[TEN]:
        if if_skip:
            return None

        action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
        obj_alpha = (self.alpha_log * (-logprob + self.target_entropy).detach()).mean()
        self.optimizer_update(self.alpha_optim, obj_alpha)

        alpha = self.alpha_log.exp().detach()
        obj_actor = (self.cri(state, action_pg) - logprob * alpha).mean()
        return obj_actor


class PendulumEnv(gym.Wrapper):  # a demo of custom env
    def __init__(self):
        gym_env_name = 'Pendulum-v1'
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self, **kwargs) -> Tuple[ARY, dict]:  # reset the agent in env
        state, info_dict = self.env.reset()
        return state, info_dict

    def step(self, action: ARY) -> Tuple[ARY, float, bool, bool, dict]:  # agent interacts in env
        # OpenAI Pendulum env set its action space as (-2, +2). It is bad.
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, terminated, truncated, info_dict = self.env.step(action * 2)
        state = state.reshape(self.state_dim)
        return state, float(reward) * 0.5, terminated, truncated, info_dict


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = list()

    def evaluate_and_save(self, actor: ActorBase, horizon_len: int, logging_tuple: tuple, max_steps_per_episode: int):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return None
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor, max_steps_per_episode) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        th.save(actor.state_dict(), f"{self.cwd}/actor.pth")

        return avg_r


def get_rewards_and_steps(env, actor: ActorBase, max_steps_per_episode: int, if_render: bool = False):
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(max_steps_per_episode):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if terminated or truncated:
            break
    cumulative_returns = getattr(env.unwrapped, 'cumulative_returns', cumulative_returns)
    return cumulative_returns, episode_steps + 1


def valid_agent(env_class, env_args: dict, net_dims: List[int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, max_steps_per_episode=500, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")