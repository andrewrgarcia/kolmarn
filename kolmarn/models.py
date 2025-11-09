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
                 basis="kan_spline", composition="sum"):
        super().__init__()

        # normalize composition to list (one per layer)
        if isinstance(composition, str):
            composition = [composition] * len(layer_sizes)
        elif len(composition) != len(layer_sizes):
            raise ValueError("composition length must match number of layers")

        layers = []
        prev = in_features
        for i, out in enumerate(layer_sizes):
            layers.append(
                KANLayer(
                    in_features=prev,
                    out_features=out,
                    num_basis=num_basis,
                    knots_trainable=knots_trainable,
                    adaptive_knots=adaptive_knots,
                    basis=basis,
                    composition=composition[i]
                )
            )
            prev = out

        self.layers = nn.ModuleList(layers)

    @classmethod
    def from_layers(cls, layers):
        """
        Build a KANSequential model directly from a list of KANLayer objects.
        This bypasses the size/in_features constructor.
        """
        model = cls.__new__(cls)   # create uninitialized instance
        nn.Module.__init__(model)  # manually initialize nn.Module

        model.layers = nn.ModuleList(layers)
        return model

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
