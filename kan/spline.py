import torch
import torch.nn as nn


class BSplineBasis(nn.Module):
    """Cubic B-spline radial basis approximation for each input feature."""
    def __init__(self, num_basis, knots_trainable=False):
        super().__init__()
        self.num_basis = num_basis
        knots = torch.linspace(0, 1, num_basis)
        self.knots = nn.Parameter(knots, requires_grad=knots_trainable)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, F, 1)
        knots = self.knots.view(1, 1, -1)  # (1, 1, K)
        distance = torch.abs(x - knots)
        B = (1 - distance ** 3).clamp(min=0)    # clamp is activation
        return B

class RBFBasis(nn.Module):
    """Gaussian radial basis expansion."""
    def __init__(self, num_basis, knots_trainable=False, sigma=0.2):
        super().__init__()
        self.num_basis = num_basis
        knots = torch.linspace(0, 1, num_basis)
        self.knots = nn.Parameter(knots, requires_grad=knots_trainable)
        self.sigma = sigma

    def forward(self, x):
        x = x.unsqueeze(-1)
        d = (x - self.knots.view(1, 1, -1)) ** 2
        return torch.exp(-d / (2 * self.sigma ** 2))
