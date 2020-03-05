from .network_ensemble import NetworkEnsemble

def get_network(network_args):
    if network_args is None:
        return None
    network_args = network_args.copy()
    if 'separate_networks' in network_args:
        separate_networks = network_args.pop('separate_networks')
    else:
        separate_networks = False
    network_type = network_args.pop('type')
    network_type = network_type.lower()
    if network_type == 'fully_connected':
        from .fully_connected import FullyConnectedNetwork
        network = FullyConnectedNetwork(**network_args)
    elif network_type == 'convolutional':
        from .convolutional import ConvolutionalNetwork
        network = ConvolutionalNetwork(**network_args)
    elif network_type == 'recurrent':
        from .recurrent import RecurrentNetwork
        network = RecurrentNetwork(**network_args)
    elif network_type == 'ar_fully_connected':
        from .ar_fully_connected import ARFullyConnectedNetwork
        network = ARFullyConnectedNetwork(**network_args)
    else:
        raise NotImplementedError

    n_networks = 2 if separate_networks else 1
    return NetworkEnsemble(network, n_networks)
