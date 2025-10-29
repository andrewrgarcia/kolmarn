import torch
import torch.nn as nn
from kolmarn.layers import KANLayer


class KANSequential(nn.Module):
    """Stack of KAN layers."""
    def __init__(self, in_features, layer_sizes, num_basis=10,
                 knots_trainable=False):
        super().__init__()
        layers = []
        prev = in_features

        for out in layer_sizes:
            layers.append(
                KANLayer(prev, out, num_basis=num_basis,
                         knots_trainable=knots_trainable)
            )
            prev = out

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    model = KANSequential(3, [8, 1], num_basis=12)
    x = torch.rand(5, 3)
    print(model(x).shape)  # torch.Size([5, 1])
