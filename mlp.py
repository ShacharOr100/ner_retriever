from __future__ import annotations
from dataclasses import dataclass
from torch.nn import Module, Linear, SiLU, Sequential, Dropout
import torch

@dataclass
class MLPArgs:
    input_layer: int = 1024
    hidden_layer: int = 500
    output_layer: int = 500
    dropout: float = 0.1


class MLP(Module):

    def __init__(self, args: MLPArgs=None):
        super().__init__()
        args = args or MLPArgs()
        input_layer = Linear(args.input_layer, args.hidden_layer)
        activation = SiLU()
        noise = Dropout(args.dropout)
        output_layer = Linear(args.hidden_layer, args.output_layer)
        self.net = Sequential(input_layer, activation, noise, output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)