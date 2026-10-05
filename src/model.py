"""Model architecture, verified against architecture.py (PRD Section 5)."""
import torch
from torch import nn


class MultipleRegression(nn.Module):
    def __init__(self, num_features, num_units_1=15, num_units_2=5,
                 activation=nn.Tanh, dropout_rate=0):
        super().__init__()
        self.layer_1 = nn.Linear(num_features, num_units_1)
        self.layer_2 = nn.Linear(num_units_1, num_units_2)
        self.layer_out = nn.Linear(num_units_2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = activation()
        for layer in (self.layer_1, self.layer_2, self.layer_out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dropout(self.act(self.layer_1(x)))
        x = self.dropout(self.act(self.layer_2(x)))
        return torch.exp(self.layer_out(x))


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
