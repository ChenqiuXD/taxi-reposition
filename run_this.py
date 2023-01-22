import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
from utils.env_runner import EnvRunner
from utils.make_env import make_env
from environment.config import get_config
from utils.recorder import Recorder
from utils.get_output_folder import get_output_folder

def parse_args(args, parser):
    """ Add arguments to the 'args' variable """
    parser.add_argument('--epsilon', type=float, default=0.9, help="Epsilon greedy")
    parser.add_argument('--tau', type=float, default=1e-3, help="Learning rate for the soft update")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--mode', type=str, default="train", help='train, test')
    parser.add_argument('--continue_num', type=int, default=0, help='if mode is continue, then load network parameters from this run_(continue_num)')
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
    # torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Get save path
    if all_args.mode == 'train':
        all_args.output_path = get_output_folder(all_args.algorithm_name, mode=all_args.mode, continue_num=all_args.continue_num)
    elif all_args.mode == 'test':
        # TODO: how to test our algorithm? 
        pass

    # create env
    env = make_env(all_args)
    recorder = Recorder(all_args)
    recorder.record_init_settings(env.get_init_setting())

    # Choose algorithm to use
    print("Using policy: ", all_args.algorithm_name)
    runner = EnvRunner(all_args, env, recorder)

    # Train the agent
    # reward_result_list = []
    if all_args.mode=="train":  # for training agents
        total_num_steps = 0
        
        runner.warmup() # Warmup drivers with all minimum bonuses. 
        while total_num_steps < all_args.num_env_steps:
            # TODO: now we do not use tensorboard to record, please add this funciton later. 
            reward_list = runner.run()  # Run an episode 
            total_num_steps += 1
            print("---------------------------------------------------------------------------------------------------")
            print("At episode ", total_num_steps, " reward sum is: ", np.sum([reward_list[i][-1] for i in range(len(reward_list))]))
            print("---------------------------------------------------------------------------------------------------\n\n\n")

            # Save data amid process
            if total_num_steps % int(all_args.num_env_steps / 5)==0:
                runner.store_data()
        runner.store_data()
    elif all_args.mode=="test":
        pass
    else:
        raise RuntimeError("Unexpected mode name '{}'. Allowed mode names are: 'train', 'test'. ".format(all_args.mode))


if __name__ == "__main__":
    # Options: null, heuristic, direct, ddpg(TODO), metaGrad(Proposed, TODO)
    algo = 'ddpg'

    # Recommended parameters:
    # episode_length: 1/6;          lr_drivers: 5e-3;           warmup_steps: 3000;             num_env_steps: 10000-20000 (depends on episode_length)
    # lr: 1e-3(direct), 1e-4 (ddpg);            tau: 5e-3;          batch_size: 16
    # buffer_size: 128(should be small, since lower-level agents change policies);   
    input_args = ['--algorithm_name', algo, '--seed', '35', '--mode', 'train',  
                  '--episode_length', '1', '--min_bonus', '0', '--max_bonus', '4', '--lr_drivers', '5e-3',
                  '--warmup_steps', '3000', '--num_env_steps', '50000', 
                  '--lr', '1e-3', '--tau', '5e-3', '--buffer_size', '128', '--batch_size', '10']

    # Check if there are input from system, then run the command.
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        main(input_args)
