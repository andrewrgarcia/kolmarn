import torch
import torch.nn as nn
from kan.spline import cubic_bspline_basis


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with spline basis expansion."""
    def __init__(self, in_features, out_features, num_basis=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        
        self.knots = nn.Parameter(torch.linspace(0, 1, num_basis))
        self.coeff = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (batch, in_features)
        B = cubic_bspline_basis(x, self.knots)  # (B, F, K)

        # Apply coeffs: sum over in_features and basis
        # (B, F, K) @ (out, F, K) -> (B, out)
        spline_term = torch.einsum("bfk,ofk -> bo", B, self.coeff)


if __name__ == "__main__":
    layer = KANLayer(3, 2, num_basis=16)
    x = torch.rand(5, 3)
    y = layer(x)
    print("Output shape:", y.shape)
    print("Output:", y)
