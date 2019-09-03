def get_network(network_args):
    if network_args is None:
        return None
    network_args = network_args.copy()
    network_type = network_args.pop('type')
    network_type = network_type.lower()
    if network_type == 'fully_connected':
        from .fully_connected import FullyConnectedNetwork
        return FullyConnectedNetwork(**network_args)
    elif network_type == 'convolutional':
        from .convolutional import ConvolutionalNetwork
        return ConvolutionalNetwork(**network_args)
    elif network_type == 'recurrent':
        from .recurrent import RecurrentNetwork
        return RecurrentNetwork(**network_args)
    else:
        raise NotImplementedError
