import torch.nn as nn
from types import SimpleNamespace
import torch as th
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Currently just boilerplate code, to tweak architecture
    """

    def __init__(
        self,
        input_shape: int,
        hidden_size1: int,
        hidden_size2: int,
        output_size: int,
        args: SimpleNamespace,
    ):
        """
        Input to the network will be of shape [B, n_agents, obs_size]
        """
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.device = th.device(args.device)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        # return q-values for actions
        return q
