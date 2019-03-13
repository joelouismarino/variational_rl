def get_network(network_args):
    if network_args is None:
        return None
    network_type = network_args['type'].lower()
    del network_args['type']
    if network_type == 'fully_connected':
        from .fully_connected import FullyConnectedNetwork
        return FullyConnectedNetwork(**network_args)
    elif network_type == 'convolutional':
        from .convolutional import ConvolutionalNetwork
        return ConvolutionalNetwork(**network_args)
    elif network_type == 'recurrent':
        from .recurrent import RecurrentNetwork
        return RecurrentNetwork(**network_args)
    elif network_type == 'vizdoom_encoder':
        from .vizdoom import ConvEncoder
        return ConvEncoder(**network_args)
    elif network_type == 'vizdoom_decoder':
        from .vizdoom import ConvDecoder
        return ConvDecoder(**network_args)
    elif network_type == 'vizdoom_skip_encoder':
        from .vizdoom import SkipConvEncoder
        return SkipConvEncoder(**network_args)
    elif network_type == 'vizdoom_skip_decoder':
        from .vizdoom import SkipConvDecoder
        return SkipConvDecoder(**network_args)
    # elif network_type == 'minigrid_discriminative':
    #     from .minigrid import DiscriminativeEncoder
    #     return DiscriminativeEncoder(**network_args)
    elif network_type == 'minigrid_conv':
        from .minigrid import MiniGridConv
        return MiniGridConv(**network_args)
    else:
        raise NotImplementedError
