from ..network import Network
from ...layers import ConvolutionalLayer


class ConvEncoder(Network):
    """
    Convolutional encoder model from Ha & Schmidhuber, 2018.
    """
    def __init__(self):
        super(ConvEncoder, self).__init__(n_layers=4)
        self.layers[0] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=4,
                                            padding=0, stride=2, non_linearity='relu')
        self.layers[1] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=4,
                                            padding=0, stride=2, non_linearity='relu')
        self.layers[2] = ConvolutionalLayer(n_input=64, n_output=128, filter_size=4,
                                            padding=0, stride=2, non_linearity='relu')
        self.layers[3] = ConvolutionalLayer(n_input=128, n_output=256, filter_size=4,
                                            padding=0, stride=2, non_linearity='relu')
        self.n_out = 1024

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = output.view(-1, 1024)
        return output
