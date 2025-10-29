import torch
import torch.nn as nn
from kan.spline import BSplineBasis


class KANLayer(nn.Module):
    """Kolmogorov-Arnold layer with spline basis expansion."""
    def __init__(self, in_features, out_features, num_basis=10,
                 knots_trainable=False):
        super().__init__()

        self.basis = BSplineBasis(num_basis, knots_trainable)
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