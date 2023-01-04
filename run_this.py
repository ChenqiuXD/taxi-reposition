import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
from env_runner import EnvRunner
from make_env import make_env
from environment.config import get_config
from recorder import Recorder

def parse_args(args, parser):
    parser.add_argument('--epsilon', type=int, default=0.9,
                        help="Epsilon greedy")
    parser.add_argument('--train', action="store_false", default=True, help='Training or evaluation')
    parser.add_argument('--max_warmup_steps', type=int, default=100, help='drivers warmup episodes')
    parser.add_argument('--render', action="store_false", default=True, help='Render or not')
    parser.add_argument('--continue_training', action="store_true", default=False, help='Whether load last iteration and continue training')
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

    # create env
    env = make_env(all_args)
    recorder = Recorder(all_args)
    recorder.record_init_settings(env.get_setting())

    # Choose algorithm to use
    print("Using policy: ", all_args.algorithm_name)
    runner = EnvRunner(all_args, env)

    # Train the agent
    reward_result_list = []
    if all_args.train:  # for training agents
        # if all_args.continue_training:
        #     try:
        #         # !!!!!! PLEASE change the "last_step_cnt" variable in env_runner.py, the 'restore' function
        #         #  so that the runner.restore() can find file. 
        #         runner.restore()
        #     except:
        #         print("Unable to restore, try training from scratch")
        total_num_steps = 0
        
        # warmup to fill the buffer
        # runner.warmup()
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
        recorder.store_data()
    else:   # for evaluating agents
        pass
        # runner.restore(isEval=True)
        # reward_result_list = np.zeros(all_args.num_env_steps)
        # for i in range(all_args.num_env_steps):
        #     reward_list = runner.eval()
        #     reward_result_list[i] = np.mean(reward_list)
        #     print("During evaluation iteration ", str(i), " the mean reward is: ", np.mean(reward_list))
        # print("Final result: the average mean reward over ", all_args.num_env_steps, " is ", np.mean(reward_result_list))


if __name__ == "__main__":
    # Options: null, random, heuristic, DDPG, metaGrad, direct
    algo = 'heuristic'

    input_args = ['--algorithm_name', algo, '--seed', '35', '--episode_length', '6', '--lr', '1e-4', '--buffer_size', '16',
                  '--batch_size', '4', '--hard_update_interval', '20', '--max_warmup_steps', '200', '--num_env_steps', '5000', '--render']

    # Check if there are input from system, then run the command.
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        main(input_args)
