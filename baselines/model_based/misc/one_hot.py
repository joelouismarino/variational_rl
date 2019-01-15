import torch


def one_hot_to_index(input):
    input_shape = input.shape
    device = input.device
    if len(input_shape) == 1:
        return (torch.arange(input_shape).to(torch.float32).to(device) * input).sum()
    else:
        return (torch.arange(input_shape[1]).to(torch.float32).to(device) * input).sum(dim=1)

def index_to_one_hot(input):
    pass
