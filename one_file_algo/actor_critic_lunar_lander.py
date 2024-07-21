import os
import sys
import time

import numpy as np
import torch as th
import gym

ARY = np.ndarray
TEN = th.Tensor

from utils_for_actor_critic import *




def train_agent(args: Config):
    args.init_before_training()
    th.set_grad_enabled(False)

    evaluator = Evaluator(
        eval_env=build_env(args.env_class, args.env_args),
        eval_per_step=args.eval_per_step,
        eval_times=args.eval_times,
        cwd=args.cwd,
    )

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.last_state, info_dict = env.reset()

    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
    )
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer

    agent.update_net(buffer, if_skip_actor=False)
    while True:  # start training
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        logging_tuple = agent.update_net(buffer)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


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
        # print("\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
        #       "\n| `time`: Time spent from the start of training to this moment."
        #       "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
        #       "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
        #       "\n| `avgS`: Average of steps in an episode."
        #       "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
        #       "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
        #       f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, actor: ActorBase, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        th.save(actor.state_dict(), f"{self.cwd}/actor.pth")

        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def get_rewards_and_steps(env, actor: ActorBase, if_render: bool = False):
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
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
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")

def train_sac_td3_ddpg_for_lunar_lander(gpu_id: int = 0, drl_id: int = 0):
    agent_class = [AgentSAC, AgentTD3, AgentDDPG][drl_id]  # DRL algorithm name
    print(f"agent_class {agent_class.__name__}")

    env_class = gym.make
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',  # A lander learns to land on a landing pad
        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 2,  # fire main engine or side engine.
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.horizon_len = 256  # collect horizon_len step while exploring, then update network
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.state_update_tau = 1e-2  # do rolling normalization on state using soft update tau
    args.batch_size = 256  # do rolling normalization on state using soft update tau
    args.gamma = 0.98

    args.eval_times = 32
    args.eval_per_step = int(2e4)

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:"):
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)



if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DRL_ID = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    train_sac_td3_ddpg_for_lunar_lander(gpu_id=GPU_ID, drl_id=DRL_ID)