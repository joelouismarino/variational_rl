def get_model(type, variable, dist, network_args):
    """
    Function to create each model.

    Args:
        type (str): 'discriminative' or 'generative'
        variable (str): 'state', 'action', 'observation', 'reward', or 'done'
        dist (str): type of distribution, e.g. 'prior'
        network_args (dict): arguments to create the network
    """
    if network_args is None:
        return None
    if type == 'discriminative':
        if variable == 'state':
            if dist in ['prior', 'inference']:
                # same form for both distributions
                from .discriminative.state_prior import StatePrior
                return StatePrior(network_args)
        elif variable == 'action':
            if dist in ['prior', 'inference']:
                # same form for both distributions
                from .discriminative.action_prior import ActionPrior
                return ActionPrior(network_args)
    elif type == 'generative':
        if variable == 'state':
            if dist == 'prior':
                pass
            elif dist == 'inference':
                pass
        elif variable == 'action':
            if dist == 'prior':
                pass
            elif dist == 'inference':
                pass
        elif variable == 'observation':
            if dist == 'likelihood':
                pass
        elif variable == 'reward':
            if dist == 'likelihood':
                pass
        elif variable == 'done':
            if dist == 'likelihood':
                pass
