import torch
import torch.nn as nn
from src.distributions.flows import BatchNormTransform

class Model(nn.Module):
    """ Constructor for nn models with device property """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @property
    def device(self):
        return next(self.parameters()).device


class MLP(Model):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, activation, batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.batch_norm = batch_norm
        
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden):
            layers.append(act)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

        if batch_norm:
            self.bn = BatchNormTransform(input_dim, affine=False)

    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, hidden_dim={}, num_hidden={}, activation={}, batch_norm={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, 
            self.hidden_dim, self.num_hidden, self.activation, self.batch_norm
        )
        return s

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
            
        for layer in self.layers:
            x = layer(x)
        return x