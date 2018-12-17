def clear_gradients(params):
    for param in params:
        param.grad = None
