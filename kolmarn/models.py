import torch
import torch.nn as nn
from kolmarn.layers import KANLayer


class KANSequential(nn.Module):
    """
    Stack of KAN layers.
    Supports adaptive knot propagation and multiple basis types.
    """
    def __init__(self, in_features, layer_sizes, num_basis=10,
                 knots_trainable=False, adaptive_knots=False,
                 basis="kan_spline"):
        super().__init__()
        layers = []
        prev = in_features

        for out in layer_sizes:
            layers.append(
                KANLayer(
                    in_features=prev,
                    out_features=out,
                    num_basis=num_basis,
                    knots_trainable=knots_trainable,
                    adaptive_knots=adaptive_knots,
                    basis=basis
                )
            )
            prev = out

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = KANSequential(3, [8, 1], num_basis=12, adaptive_knots=True)
    x = torch.rand(5, 3)
    y = model(x)
    print("Output shape:", y.shape)
    print("Knots from first layer:", model.layers[0].basis.compute_knots().detach().numpy())
