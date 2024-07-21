import os
import sys
import numpy as np
import torch as th
from tqdm import tqdm

ARY = np.ndarray
TEN = th.Tensor

from utils_for_actor_critic_target import *


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
        self.state_update_tau = 1e-2  # 1e-1 ~ 1e-6
        self.batch_size = int(64)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(256)  # collect horizon_len step while exploring, then update network
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

        self.eval_times = int(16)  # number of times that get episodic cumulative return
        self.eval_per_step = int(1e2)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_target'
        os.makedirs(self.cwd, exist_ok=True)

def train_agent(args: Config, max_episodes: int, max_steps_per_episode: int, repeat_times: int):
    score_traj = []

    for repeat in tqdm(range(repeat_times)):
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

        scores = []
        for episode in tqdm(range(max_episodes), leave=False):  # start training
            buffer_items = agent.explore_env(env, args.horizon_len)
            buffer.update(buffer_items)

            logging_tuple = agent.update_net(buffer)

            # Evaluate and save cumulative reward
            cumulative_reward = evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple, max_steps_per_episode)
            if cumulative_reward != None:
                scores.append(cumulative_reward)
            else:
                print("Now accumulated reward is None")
        score_traj.append(scores)

    return score_traj

def train_sac_td3_ddpg_for_pendulum(gpu_id: int = 0, drl_id: int = 0, repeat_times: int = 2, max_episodes: int = 10000, max_steps_per_episode: int = 500):
    agent_class = [AgentSAC, AgentTD3, AgentDDPG][drl_id]  # DRL algorithm name
    print(f"agent_class {agent_class.__name__}")

    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum-v1',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)  # see `Config` for explanation
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    scores = train_agent(args, max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode, repeat_times=repeat_times)
    np.save(f'{args.cwd}/scores.npy', scores)
    # if input("| Press 'y' to load actor.pth and render:"):
    #     actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
    #     actor_path = f"{args.cwd}/{actor_name}"
    #     valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DRL_ID = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    train_sac_td3_ddpg_for_pendulum(gpu_id=GPU_ID, drl_id=DRL_ID, max_episodes=60, repeat_times=2)