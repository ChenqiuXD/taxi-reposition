import sys

import numpy as np
import torch
from make_env import make_env
from utils.runner import Runner
from utils.config import get_config


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--epsilon', type=int, default=0.1,
                        help="Epsilon greedy")
    parser.add_argument('--train', action="store_false", default=True, help='Training or evaluation')
    parser.add_argument('--render', action="store_false", default=True, help='Render or not')
    parser.add_argument('--continue_training', action="store_true", default=True, help='Whether load last iteration and continue training')
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
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_env(all_args.scenario_name)
    env.discrete_action_input = True

    # Choose algorithm to use
    if all_args.algorithm_name in ['qmix', 'iql', 'vdn']:
        print("Using policy: ", all_args.algorithm_name)
        runner = Runner(all_args, env)
    else:
        raise NotImplementedError("ERROR: algorithm not implemented. Possible policies: qmix, vdn, ddpg")

    # Train the agent
    reward_result_list = []
    if all_args.train:  # for training agents
        if all_args.continue_training:
            try:
                runner.restore()
            except:
                print("Unable to restore the network parameters, try training from scratch")
        total_num_steps = 0
        while total_num_steps < all_args.num_env_steps:
            reward_list = runner.run()
            total_num_steps += 1
            print("At episode ", total_num_steps, " reward sum is: ", np.sum(reward_list))
            reward_result_list.append(np.sum(reward_list))
        result_name = "result_"+all_args.algorithm_name+"_"+str(all_args.num_env_steps)+".npy"
        np.save(result_name, np.array(reward_result_list))
    else:   # for evaluating agents
        runner.restore()
        reward_result_list = np.zeros(all_args.num_env_steps)
        for i in range(all_args.num_env_steps):
            reward_list = runner.eval()
            reward_result_list[i] = np.mean(reward_list)
            print("During evaluation iteration ", str(i), " the mean reward is: ", np.mean(reward_list))
        print("Final result: the average mean reward over ", all_args.num_env_steps, " is ", np.mean(reward_result_list))
    return reward_result_list


if __name__ == "__main__":
    algo = 'qmix'

    # For eval
    # input_args = ['--algorithm_name', algo, '--seed', '50', '--episode_length', '100',
    #               '--num_env_steps', '20', '--train']

    # For train
    input_args = ['--env_name', 'MPE', '--algorithm_name', algo,
                  '--seed', '10', '--episode_length', '100', '--lr', '1e-4',
                  '--hard_update_interval', '500', '--num_env_steps', '10000', '--render']

    # Check if there are input from system, then run the command.
    if sys.argv[1:]:
        main(sys.argv[1:])
    else:
        # test_list = np.zeros(5)
        # for seed in range(5):
        #     input_args[3] = str(seed)
        #     test_list[seed] = np.mean(main(input_args))
        # print("Final result: ", np.mean(test_list))
        main(input_args)
