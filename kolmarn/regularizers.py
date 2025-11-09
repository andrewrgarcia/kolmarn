import torch
import torch.nn.functional as F

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


def magnitude_entropy_penalty(model, lambda_mag=1e-3, lambda_ent=1e-3):
    mag_loss, ent_loss = 0.0, 0.0

    for layer in model.modules():
        if hasattr(layer, "coeff"):
            A = torch.abs(layer.coeff)
            mag_loss += A.mean()

            # row/col probabilities for entropy
            P_row = A.sum(dim=-1)
            P_row = P_row / (P_row.sum() + 1e-8)
            P_col = A.sum(dim=0)
            P_col = P_col / (P_col.sum() + 1e-8)
            ent_loss += -(P_row * (P_row + 1e-8).log()).sum()
            ent_loss += -(P_col * (P_col + 1e-8).log()).sum()

    return lambda_mag * mag_loss + lambda_ent * ent_loss
