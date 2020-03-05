import torch


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self):
        torch.distributions.Transform.__init__(self)
        torch.nn.Module.__init__(self)

    def __hash__(self):
        return torch.nn.Module.__hash__(self)
