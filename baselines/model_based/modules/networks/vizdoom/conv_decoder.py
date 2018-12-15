from ..network import Network
from ...layers import FullyConnectedLayer, TransposedConvLayer


class ConvDecoder(Network):
    """
    Convolutional decoder model from Ha & Schmidhuber, 2018.
    """
    def __init__(self, n_input=64):
        super(ConvDecoder, self).__init__(n_layers=4)
        self.layers[0] = FullyConnectedLayer(n_input=n_input, n_output=1024)
        self.layers[1] = TransposedConvLayer(n_input=1024, n_output=128, filter_size=5,
                                             padding=0, stride=2, non_linearity='relu')
        self.layers[2] = TransposedConvLayer(n_input=128, n_output=64, filter_size=5,
                                             padding=0, stride=2, non_linearity='relu')
        self.layers[3] = TransposedConvLayer(n_input=64, n_output=32, filter_size=6,
                                             padding=0, stride=2, non_linearity='relu')
        self.n_out=32

    def forward(self, input):
        output = input
        output = self.layers[0](output)
        output = output.view(-1, 1024, 1, 1)
        for layer in self.layers[1:]:
            output = layer(output)
        return output
