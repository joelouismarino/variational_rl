def get_variable(type, args):
    if type == 'latent':
        return get_latent_variable(args)
    elif type == 'observed':
        return get_observed_variable(args)
    elif type == 'value':
        from .value_variable import ValueVariable
        return ValueVariable(**args)
    else:
        raise NotImplementedError


def get_latent_variable(variable_args):
    variable_type = variable_args['type'].lower()
    del variable_args['type']
    if variable_type == 'fully_connected':
        from .latent_variables import FullyConnectedLatentVariable
        return FullyConnectedLatentVariable(**variable_args)
    elif variable_type == 'convolutional':
        from .latent_variables import ConvolutionalLatentVariable
        return ConvolutionalLatentVariable(**variable_args)
    else:
        raise NotImplementedError


def get_observed_variable(variable_args):
    variable_type = variable_args['type'].lower()
    del variable_args['type']
    if variable_type == 'fully_connected':
        from .observed_variables import FullyConnectedObservedVariable
        return FullyConnectedObservedVariable(**variable_args)
    elif variable_type == 'convolutional':
        from .observed_variables import ConvolutionalObservedVariable
        return ConvolutionalObservedVariable(**variable_args)
    elif variable_type == 'transposed_conv':
        from .observed_variables import TransposedConvObservedVariable
        return TransposedConvObservedVariable(**variable_args)
    else:
        raise NotImplementedError
