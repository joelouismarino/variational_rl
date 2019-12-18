from .direct import DirectEstimator
from .model_based import ModelBasedEstimator

def get_q_value_estimator(kwargs):
    estimator_type = kwargs.pop('estimator_type')
    if estimator_type == 'direct':
        return DirectEstimator(**kwargs)
    elif estimator_type == 'model_based':
        return ModelBasedEstimator(**kwargs)
    else:
        raise NotImplementedError
