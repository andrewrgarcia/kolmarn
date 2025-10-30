import torch 

def spline_smoothness_penalty(model):
    """Second-order finite differences of spline coeffs."""
    loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "coeff"):
            c = layer.coeff  # (out, in, K)
            diff2 = c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]
            loss = loss + (diff2 ** 2).mean()
    return loss


def knot_spacing_penalty(model, target_spacing=None):
    """Encourage knots to remain well-distributed and monotonic."""
    penalty = 0.0
    for layer in model.modules():
        if hasattr(layer, "compute_knots"):
            knots = layer.compute_knots()
            diffs = torch.diff(knots)

            spacing = diffs / (diffs.mean().detach() + 1e-8)
            penalty += ((spacing - 1) ** 2).mean()

            if target_spacing is not None:
                penalty += ((diffs.mean() - target_spacing) ** 2)
    return penalty
