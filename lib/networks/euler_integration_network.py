import torch
from .network import Network
from ..layers import FullyConnectedLayer


# TODO: disjoint networks, angle loss

class EulerIntegrationNetwork(Network):
    """
    A network with Euler integration as an inductive bias.
    """
    def __init__(self, n_layers, n_input, n_units, num_state_dims,
                 num_velocity_dims, num_action_dims, angle_indices, dt,
                 batch_norm=False, layer_norm=True, non_linearity='leaky_relu',
                 dropout=None):
        super().__init__(n_layers, None)

        self.num_state_dims = num_state_dims
        self.num_velocity_dims = num_velocity_dims
        self.num_action_dims = num_action_dims
        self.angle_indices = angle_indices
        self.dt = dt
        assert (self.num_state_dims + self.num_action_dims) == n_input

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]
        else:
            assert len(n_units) == n_layers

        if type(non_linearity) == str:
            non_linearity = [non_linearity for _ in range(n_layers)]
        else:
            assert len(non_linearity) == n_layers

        if type(batch_norm) == bool:
            batch_norm = [batch_norm for _ in range(n_layers)]
        else:
            assert len(batch_norm) == n_layers

        if type(layer_norm) == bool:
            layer_norm = [layer_norm for _ in range(n_layers)]
        else:
            assert len(layer_norm) == n_layers

        if type(dropout) == float or dropout is None:
            dropout = [dropout for _ in range(n_layers)]
        else:
            assert len(dropout) == n_layers

        n_in = n_input
        for l in range(n_layers):
            self.layers[l] = FullyConnectedLayer(n_in, n_units[l],
                                                 batch_norm=batch_norm[l],
                                                 layer_norm=layer_norm[l],
                                                 non_linearity=non_linearity[l],
                                                 dropout=dropout[l])
            n_in = n_units[l]
        self.n_out = n_in

    def forward(self, input):
        for ind, layer in enumerate(self.layers):
            if ind == 0:
                preds = layer(input)
            else:
                preds = layer(preds)

        preds = input[..., :-self.num_action_dims] + preds

        preds[..., :-self.num_velocity_dims] += \
            self.dt * preds[..., -self.num_velocity_dims:].detach()

        angle_pred_clone = preds[:, self.angle_indices].clone()
        preds[:, self.angle_indices] = torch.atan2(torch.sin(angle_pred_clone),
                                                   torch.cos(angle_pred_clone))
        return preds
