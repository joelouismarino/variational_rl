import copy
from .direct_q import DirectQEstimator
from .model_based_q import ModelBasedQEstimator
from .direct_v import DirectVEstimator
from .simulator_q import SimulatorQEstimator


def get_value_estimator(state_action, kwargs):
    """
    Constructs a value estimator.

    Args:
        state_action (str): either 'state' or 'action'
        kwargs (dict): the estimator keyword arguments
    """
    assert state_action in ['state', 'action']
    new_kwargs = copy.deepcopy(kwargs)
    if state_action == 'action':
        estimator_type = new_kwargs.pop('estimator_type')
        if estimator_type == 'direct':
            return DirectQEstimator(**new_kwargs)
        elif estimator_type == 'model_based':
            return ModelBasedQEstimator(**new_kwargs)
        elif estimator_type == 'simulator':
            return SimulatorQEstimator(**new_kwargs)
        else:
            raise NotImplementedError
    else:
        return DirectVEstimator(**new_kwargs)
