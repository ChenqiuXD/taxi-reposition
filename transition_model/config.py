import argparse


def get_config(args):
    parser = argparse.ArgumentParser(
        description="OFF-POLICY", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--epoch", type=int, default=10, help="maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--max_bonus", type=int, default=5, help="max bonus for each node")
    parser.add_argument("--converge_criterion", type=float, default=1e-3, help="Check whether policies have converged")
    parser.add_argument("--sim_steps", type=int, default=800, help="Length of simulation steps")
    parser.add_argument("--display", action="store_false", default=True, help="Whether display the training process in SUMO")

    all_args = parser.parse_known_args(args)[0]

    return all_args

