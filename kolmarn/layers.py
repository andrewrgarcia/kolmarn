import torch
import torch.nn as nn
from kolmarn.spline import BSplineBasis, RBFBasis


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold layer with configurable basis and adaptive knots.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_basis (int): Number of spline basis functions per feature.
        knots_trainable (bool): Whether to make fixed knots learnable.
        adaptive_knots (bool): Whether to use softmax-based adaptive knots.
        basis (str): Type of basis, one of {"kan_spline", "rbf"}.
    """
    def __init__(self, in_features, out_features,
                 num_basis=10, knots_trainable=False,
                 adaptive_knots=False, basis="kan_spline", composition="sum"):
        super().__init__()

        basis_map = {
            "kan_spline": BSplineBasis,
            "rbf": RBFBasis
        }
        if basis not in basis_map:
            raise ValueError(f"Unknown basis type: {basis}")

        self.basis = basis_map[basis](
            num_basis=num_basis,
            knots_trainable=knots_trainable,
            adaptive_knots=adaptive_knots
        )
        self.composition = composition 

        self.coeff = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.config = dict(
            in_features=in_features,
            out_features=out_features,
            num_basis=num_basis,
            knots_trainable=knots_trainable,
            adaptive_knots=adaptive_knots,
            basis=basis
        )

    def forward(self, x):
        B = self.basis(x)  # (B, F, K)
        # Apply coeffs: sum over in_features and basis
        # (B, F, K) @ (out, F, K) -> (B, out)
        spline_term = torch.einsum("bfk,ofk->bo", B, self.coeff)
        lin = self.linear(x) + self.bias

        if self.composition == "sum":
            return spline_term + lin
        elif self.composition == "prod":
            return spline_term * lin
        else:
            raise ValueError(f"Unknown composition: {self.composition}")


if __name__ == "__main__":
    layer = KANLayer(3, 2, num_basis=16, adaptive_knots=True, composition="prod")
    x = torch.rand(5, 3)
    y = layer(x)
    print("Output shape:", y.shape)
    print("First 5 knots:", layer.basis.compute_knots().detach().numpy()[:5])
    print("Output:", y)