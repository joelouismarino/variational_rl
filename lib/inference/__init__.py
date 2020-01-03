from .direct import DirectInferenceModel
from .iterative import IterativeInferenceModel
from .gradient import GradientBasedInference
from .cem import CEMInference
from .non_parametric import NonParametricInference

def get_inference_optimizer(agent, kwargs):
    opt_type = kwargs.pop('opt_type')
    if opt_type == 'direct':
        return DirectInferenceModel(agent, **kwargs)
    elif opt_type == 'iterative':
        return IterativeInferenceModel(agent, **kwargs)
    elif opt_type == 'gradient':
        return GradientBasedInference(agent, **kwargs)
    elif opt_type == 'cem':
        return CEMInference(agent, **kwargs)
    elif opt_type == 'non_parametric':
        return NonParametricInference(agent, **kwargs)
    else:
        raise NotImplementedError
