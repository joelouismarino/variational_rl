import torch.nn as nn


class Network(nn.Module):

    def __init__(self, n_layers):
        super(Network, self).__init__()
        self.layers = nn.ModuleList([None for _ in range(n_layers)])

    def forward(self, input):
        pass
