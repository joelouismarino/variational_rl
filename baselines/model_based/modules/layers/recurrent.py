import torch
import torch.nn as nn
from .layer import Layer


class RecurrentLayer(Layer):

    def __init__(self, n_input, n_output, dropout=None):
        super(RecurrentLayer, self).__init__()
        self.lstm = nn.LSTMCell(n_input, n_output)
        if dropout:
            self.dropout = nn.Dropout1d(dropout)
        self.initial_hidden = nn.Parameter(torch.zeros(1, n_output))
        self.initial_cell = nn.Parameter(torch.zeros(1, n_output))
        self.hidden_state = self._hidden_state = None
        self.cell_state = self._cell_state = None
        self._detach = False

    def forward(self, input):
        if self.hidden_state is None:
            # re-initialize the hidden state
            self.hidden_state = self.initial_hidden.repeat(input.shape[0], 1)
        if self.cell_state is None:
            # re-initialize the cell state
            self.cell_state = self.initial_cell.repeat(input.shape[0], 1)
        # detach the hidden and cell states if necessary
        hs = self.hidden_state.detach() if self._detach else self.hidden_state
        cs = self.cell_state.detach() if self._detach else self.cell_state
        # perform forward computation
        self._hidden_state, self._cell_state = self.lstm(input, (hs, cs))
        return self.dropout(self._hidden_state)

    def step(self):
        self.hidden_state = self._hidden_state
        self.cell_state = self._cell_state

    def reset(self):
        self.hidden_state = self._hidden_state = None
        self.cell_state = self._cell_state = None

    def detach_hidden_state(self):
        self._detach = True

    def attach_hidden_state(self):
        self._detach = False
