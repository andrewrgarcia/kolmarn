import torch
import torch.nn as nn


class BaseSplineBasis(nn.Module):
    """Base class for spline/radial bases with adaptive or trainable knots."""
    def __init__(self, num_basis, knots_trainable=False, adaptive_knots=False):
        super().__init__()
        self.num_basis = num_basis
        self.knots_trainable = knots_trainable
        self.adaptive_knots = adaptive_knots

        if adaptive_knots:
            # raw, unconstrained parameters (mapped to [0,1] sorted knots)
            self.raw_knots = nn.Parameter(torch.randn(num_basis))
        else:
            self.knots = nn.Parameter(
                torch.linspace(0, 1, num_basis),
                requires_grad=knots_trainable
            )

    def compute_knots(self):
        """Return knots ∈ [0,1], sorted and differentiable."""
        if not self.adaptive_knots:
            return self.knots
        weights = torch.softmax(self.raw_knots, dim=0)
        knots = torch.cumsum(weights, dim=0)
        knots = knots / knots[-1].detach()
        return knots

    def enable_adaptive_knots(self):
        """Dynamically activate adaptive knot training."""
        if not hasattr(self, "raw_knots"):
            self.raw_knots = nn.Parameter(torch.randn(self.num_basis))
        self.adaptive_knots = True

    def forward(self, x):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward(x)")


class BSplineBasis(BaseSplineBasis):
    """Cubic B-spline radial basis approximation."""
    def __init__(self, num_basis, knots_trainable=False, adaptive_knots=False):
        super().__init__(num_basis, knots_trainable, adaptive_knots)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, F, 1)
        knots = self.compute_knots().view(1, 1, -1)
        distance = torch.abs(x - knots)
        B = (1 - distance ** 3).clamp(min=0)
        return B


class RBFBasis(BaseSplineBasis):
    """Gaussian radial basis approximation."""
    def __init__(self, num_basis, knots_trainable=False,
                 adaptive_knots=False, sigma=0.2):
        super().__init__(num_basis, knots_trainable, adaptive_knots)
        self.sigma = sigma

    def forward(self, x):
        x = x.unsqueeze(-1)
        knots = self.compute_knots().view(1, 1, -1)
        d = (x - knots) ** 2
        return torch.exp(-d / (2 * self.sigma ** 2))
    

if __name__ == "__main__":
    for cls in [BSplineBasis, RBFBasis]:
        basis = cls(8, adaptive_knots=True)
        x = torch.linspace(0, 1, 5).unsqueeze(1)
        print(f"{cls.__name__} output:", basis(x).shape)
        print("knots:", basis.compute_knots().detach().numpy())
