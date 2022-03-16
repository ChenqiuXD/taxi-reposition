import torch.nn as nn


class AgentQFunction(nn.Module):
    """
    Inidividual agent q network (MLP)
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param config: contains information about network parameters
    """
    def __init__(self, args, config):
        super(AgentQFunction, self).__init__()
        self.device = args.device
        self.hidden_layers = config['hidden_layers']
        self.dim_obs = config['dim_obs']        # dim of observation
        self.dim_action = config['dim_action']  # num of possible actions

        assert len(self.hidden_layers) == 2, 'The hidden layer number in mixer_config should be 2'
        self.mlp = nn.Sequential(nn.Linear(self.dim_obs, self.hidden_layers[0]),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_layers[1], self.dim_action))

    def forward(self, x):
        """
        Compute q values for every action given current state
        :param x: (torch.tensor) state from which to compute a q values
        :return: q_values [dim_action*1] (torch.tensor) q values for every actions
        """
        x = x.to(self.device)

        return self.mlp(x)

