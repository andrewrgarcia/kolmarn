import torch
import torch.nn.functional as F


def cubic_bspline_basis(x, knots):
    """Cubic ReLU-based spline basis approximation."""
    x = x.unsqueeze(-1)  # (B, F, 1)
    knots = knots.reshape(1, 1, -1)  # (1, F, K)
    return F.relu(1 - torch.abs(x - knots) ** 3)    # simple smooth radial basis
