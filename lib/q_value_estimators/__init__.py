import copy
from .direct import DirectEstimator
from .model_based import ModelBasedEstimator

def get_q_value_estimator(kwargs):
    new_kwargs = copy.deepcopy(kwargs)
    estimator_type = new_kwargs.pop('estimator_type')
    if estimator_type == 'direct':
        return DirectEstimator(**new_kwargs)
    elif estimator_type == 'model_based':
        return ModelBasedEstimator(**new_kwargs)
    else:
        raise NotImplementedError
