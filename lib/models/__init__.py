from .model import Model


def get_model(network_args):
    """
    Function to create each model.

    Args:
        network_args (dict): arguments to create the network
    """
    if network_args is None:
        return None
    return Model(network_args)
