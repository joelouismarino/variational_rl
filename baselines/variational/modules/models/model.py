import torch.nn as nn


class Model(nn.Module):
    """
    Wrapper class around network to parameterize each of the conditional mappings.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.network = None

    @property
    def n_out(self):
        if self.network is not None:
            return self.network.n_out
        else:
            raise NotImplementedError

    def reset(self, batch_size=0):
        if self.network is not None:
            self.network.reset(batch_size)

    def detach_hidden_state(self):
        if self.network is not None:
            self.network.detach_hidden_state()

    def attach_hidden_state(self):
        if self.network is not None:
            self.network.attach_hidden_state()

    def forward(self):
        raise NotImplementedError
