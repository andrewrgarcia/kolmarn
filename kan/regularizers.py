def spline_smoothness_penalty(model):
    """Second-order finite differences of spline coeffs."""
    loss = 0.0
    for layer in model.modules():
        if hasattr(layer, "coeff"):
            c = layer.coeff  # (out, in, K)
            diff2 = c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]
            loss = loss + (diff2 ** 2).mean()
    return loss
