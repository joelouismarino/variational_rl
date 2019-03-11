def get_model(type, *args):
    """
    Function to create each model.

    Args:
        type (str): 'discriminative' or 'generative'
        variable (str): 'state', 'action', 'observation', 'reward', or 'done'
        dist (str): type of distribution, e.g. 'prior'
        network_args (dict): arguments to create the network
    """

    if len(args) == 1:
        network_args = args[0]
    elif len(args) == 3:
        variable, dist, network_args = args
    else:
        raise NotImplementedError

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
                from .generative.state_prior import StatePrior
                return StatePrior(network_args)
            elif dist == 'inference':
                from .generative.state_inference import StateInference
                return StateInference(network_args)
        elif variable == 'action':
            if dist == 'prior':
                from .generative.action_prior import ActionPrior
                return ActionPrior(network_args)
            elif dist == 'inference':
                from .generative.action_inference import ActionInference
                return ActionInference(network_args)
        elif variable == 'observation':
            if dist == 'likelihood':
                from .generative.observation_likelihood import ObservationLikelihood
                return ObservationLikelihood(network_args)
        elif variable == 'reward':
            if dist == 'likelihood':
                from .generative.reward_likelihood import RewardLikelihood
                return RewardLikelihood(network_args)
        elif variable == 'done':
            if dist == 'likelihood':
                from .generative.done_likelihood import DoneLikelihood
                return DoneLikelihood(network_args)
    elif type == 'value':
        from .value.value_model import ValueModel
        return ValueModel(network_args)
