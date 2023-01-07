import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
from utils.env_runner import EnvRunner
from utils.make_env import make_env
from environment.config import get_config
from utils.recorder import Recorder
from utils.get_output_folder import get_output_folder
from utils.plot import plot_result

def parse_args(args, parser):
    """ Add arguments to the 'args' variable """
    parser.add_argument('--epsilon', type=float, default=0.9, help="Epsilon greedy")
    parser.add_argument('--tau', type=float, default=1e-3, help="Learning rate for the soft update")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--mode', type=str, default="train", help='train or test')
    parser.add_argument('--warmup_steps', type=int, default=100, help='drivers warmup episodes')
    parser.add_argument('--render', action="store_false", default=True, help='Render or not')

    parser.add_argument('--lr_drivers', type=float, default=1e-3, help="learning rate for drivers")
    parser.add_argument('--min_bonus', type=float, default=0, help="The minimum bonus")
    parser.add_argument('--max_bonus', type=float, default=4, help='The maximum bonus')

    parser.add_argument('--output_path', type=str, default="./", help="The storage path automatically find by function get_output_folder")

    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    # Parse args
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current using ", all_args.device)
    print("Render is ", all_args.render)

    # set seeds
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(arser)
    all_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current using ", all_args.device)
    print("Render is ", all_args.render)

    # set seeds
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Get save path
    if all_args.mode == 'train':
        all_args.output_path = get_output_folder(all_args.algorithm_name)
    elif all_args.mode == 'test':
        pass

    # create env
    env = make_env(all_args)
    recorder = Recorder(all_args)
    recorder.record_init_settings(env.get_init_setting())

    # Choose algorithm to use
    print("Using policy: ", all_args.algorithm_name)
    runner = EnvRunner(all_args, env)

    # Train the agent
    reward_result_list = []
    if all_args.mode=="train":  # for training agents
        total_num_steps = 0
        
        # Warmup to fill the buffer
        runner.warmup()
        while total_num_steps < all_args.num_env_steps:
            reward_list, action_list = runner.run()
            total_num_steps += 1
            print("---------------------------------------------------------------------------------------------------")
            print("At episode ", total_num_steps, " reward sum is: ", np.sum([reward_list[i][-1] for i in range(len(reward_list))]))
            print("---------------------------------------------------------------------------------------------------\n\n\n")
            reward_result_list.append(np.sum(reward_list))

            nodes_actions = env.get_nodes_actions()
            idle_drivers = [env.games[i].get_state() for i in range(len(env.games))]
            recorder.record(reward_list, action_list, nodes_actions, idle_drivers)
        data = recorder.store_data()
        plot_result(all_args, data)
        runner.store_data()
    elif all_args.mode=="test":
        pass
    else:
        raise RuntimeError("Unexpected mode name '{}'. Allowed mode names are: 'train', 'test'. ".format(all_args.mode))


if __name__ == "__main__":
    # Options: null, random, heuristic, DDPG, metaGrad, direct
    algo = 'ddpg'

    input_args = ['--algorithm_name', algo, '--seed', '35', '--mode', 'train', 
                  '--episode_length', '6', '--min_bonus', '0', '--max_bonus', '5', '--lr_drivers', '1e-3',
                  '--warmup_steps', '2000', '--num_env_steps', '10000', 
                  '--lr', '1e-4', '--tau', '1e-3', '--buffer_size', '16', '--batch_size', '4', 
                  '--render']

    # Check if there are input from system, then run the command.
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        main(input_args)
