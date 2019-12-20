from .direct import DirectInferenceModel
from .iterative import IterativeInferenceModel
from .gradient_based import GradientBasedInference
from .cem import CEMInference
from .non_parametric import NonParametricInference

def get_inference_optimizer(kwargs):
    opt_type = kwargs.pop('opt_type')
    if opt_type == 'direct':
        return DirectInferenceModel(**kwargs)
    elif opt_type == 'iterative':
        return IterativeInferenceModel(**kwargs)
    elif opt_type == 'gradient_based':
        return GradientBasedInference(**kwargs)
    elif opt_type == 'cem':
        return CEMInference(**kwargs)
    elif opt_type == 'non_parametric':
        return NonParametricInference(**kwargs)
    else:
        raise NotImplementedError
