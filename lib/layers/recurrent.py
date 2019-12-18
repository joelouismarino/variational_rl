import torch
import torch.nn as nn
from .layer import Layer


class RecurrentLayer(Layer):
    """
    An LSTM layer.
    """
    def __init__(self, n_input, n_output, dropout=None):
        super(RecurrentLayer, self).__init__()
        self.lstm = nn.LSTMCell(n_input, n_output)
        if dropout:
            self.dropout = nn.Dropout1d(dropout)
        self.initial_hidden = nn.Parameter(torch.zeros(1, n_output))
        self.initial_cell = nn.Parameter(torch.zeros(1, n_output))
        self.hidden_state = self._hidden_state = None
        self.cell_state = self._cell_state = None
        self.planning_hidden_state = self._planning_hidden_state = None
        self.planning_cell_state = self._planning_cell_state = None
        self._detach = False
        self._planning = False

    def forward(self, input):
        if self._planning:
            # detach the hidden and cell states if necessary
            hs = self.planning_hidden_state
            cs = self.planning_cell_state
            # perform forward computation
            self._planning_hidden_state, self._planning_cell_state = self.lstm(input, (hs, cs))
            return self.dropout(self._planning_hidden_state)
        else:
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
        if self._planning:
            self.planning_hidden_state = self._planning_hidden_state
            self.planning_cell_state = self._planning_cell_state
        else:
            self.hidden_state = self._hidden_state
            self.cell_state = self._cell_state

    @property
    def state(self):
        return self.hidden_state.detach() if self._detach else self.hidden_state

    def reset(self, batch_size):
        self.hidden_state = self.initial_hidden.repeat(batch_size, 1)
        self.cell_state = self.initial_cell.repeat(batch_size, 1)
        self._hidden_state = self._cell_state = None

    def planning_mode(self, batch_size):
        self.planning_hidden_state = self.hidden_state.detach().repeat(batch_size, 1)
        self.planning_cell_state = self.cell_state.detach().repeat(batch_size, 1)
        self._planning = True

    def acting_mode(self):
        self.planning_hidden_state = self.planning_cell_state = None
        self._planning = False

    def detach_hidden_state(self):
        self._detach = True

    def attach_hidden_state(self):
        self._detach = False
