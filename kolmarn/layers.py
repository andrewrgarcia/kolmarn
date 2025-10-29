import torch
import torch.nn as nn
from kolmarn.spline import BSplineBasis, RBFBasis


class KANLayer(nn.Module):
    """Kolmogorov-Arnold layer with configurable basis type."""
    def __init__(self, in_features, out_features,
                 num_basis=10, knots_trainable=False,
                 basis="kan_spline"):
        super().__init__()

        if basis == "kan_spline":
            self.basis = BSplineBasis(num_basis, knots_trainable)
        elif basis == "rbf":
            self.basis = RBFBasis(num_basis, knots_trainable)
        else:
            raise ValueError(f"Unknown basis type: {basis}")

        self.coeff = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        B = self.basis(x)  # (B, F, K)
        # Apply coeffs: sum over in_features and basis
        # (B, F, K) @ (out, F, K) -> (B, out)
        spline_term = torch.einsum("bfk,ofk->bo", B, self.coeff)

        return spline_term + self.linear(x) + self.bias


if __name__ == "__main__":
    layer = KANLayer(3, 2, num_basis=16, knots_trainable=False)
    x = torch.rand(5, 3)
    y = layer(x)
    print("Output shape:", y.shape)
    print("Output:", y)