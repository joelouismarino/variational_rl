from ..network import Network
from ...layers import ConvolutionalLayer, FullyConnectedLayer


class ConvDiscriminator(Network):
    """
    Convolutional model for a discriminative agent.
    """
    def __init__(self, non_linearity='relu', batch_norm=False):
        super(ConvDiscriminator, self).__init__(n_layers=4)
        self.layers[0] = ConvolutionalLayer(n_input=3, n_output=8, filter_size=6,
                                            padding=0, stride=3,
                                            non_linearity=non_linearity,
                                            batch_norm=batch_norm)
        self.layers[1] = ConvolutionalLayer(n_input=8, n_output=8, filter_size=3,
                                            padding=0, stride=2,
                                            non_linearity=non_linearity,
                                            batch_norm=batch_norm)
        self.layers[2] = FullyConnectedLayer(n_input=192, n_output=128,
                                             non_linearity=non_linearity,
                                             batch_norm=batch_norm)
        self.n_out = 128

    def forward(self, input):
        output = input
        output = self.layers[0](output)
        output = self.layers[1](output)
        output = output.view(-1, 192)
        output = self.layers[2](output)
        return output
