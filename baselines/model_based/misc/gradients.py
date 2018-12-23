from torch._six import inf

def clear_gradients(params):
    for param in params:
        param.grad = None

def clip_gradients(grads, clip_value):
    clip_value = float(clip_value)
    for g in filter(lambda g: g is not None, grads):
        g.data.clamp_(min=-clip_value, max=clip_value)
    # return grads

def norm_gradients(grads, max_norm, norm_type=2):
    grads = list(filter(lambda g: g is not None, grads))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in grads)
    else:
        total_norm = 0
        for g in grads:
            norm = g.data.norm(norm_type)
            total_norm += norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return total_norm

def calc_norm(grads, norm_type=2):
    grads = list(filter(lambda g: g is not None, grads))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in grads)
    else:
        total_norm = 0
        for g in grads:
            norm = g.data.norm(norm_type)
            total_norm += norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm
