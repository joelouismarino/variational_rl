import copy
from .direct import DirectInferenceModel
from .iterative import IterativeInferenceModel
from .gradient import GradientBasedInference
from .cem import CEMInference
from .non_parametric import NonParametricInference

def get_inference_optimizer(kwargs):
    new_kwargs = copy.deepcopy(kwargs)
    opt_type = new_kwargs.pop('opt_type')
    if opt_type == 'direct':
        return DirectInferenceModel(**new_kwargs)
    elif opt_type == 'iterative':
        return IterativeInferenceModel(**new_kwargs)
    elif opt_type == 'gradient':
        return GradientBasedInference(**new_kwargs)
    elif opt_type == 'cem':
        return CEMInference(**new_kwargs)
    elif opt_type == 'non_parametric':
        return NonParametricInference(**new_kwargs)
    else:
        raise NotImplementedError
