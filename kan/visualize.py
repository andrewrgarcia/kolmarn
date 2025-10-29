import torch
import matplotlib.pyplot as plt
import math


@torch.no_grad()
def plot_feature_splines(model, feature_index=0, layer_index=0, num_points=200):
    # Extract KAN layers
    kan_layers = [layer for layer in model.modules() if hasattr(layer, "coeff")]
    if not kan_layers:
        raise ValueError("No KAN layers found in model.")
    try:
        layer = kan_layers[layer_index]
    except IndexError:
        raise ValueError(f"Invalid layer_index {layer_index}. Model has {len(kan_layers)} KAN layers.")

    # Evaluate basis functions
    x = torch.linspace(0, 1, num_points).unsqueeze(1)
    B = layer.basis(x)[:, 0, :]
    coeff = layer.coeff[:, feature_index, :]
    H = coeff.shape[0]

    N = math.ceil(math.sqrt(H))

    fig, axs = plt.subplots(
        N, N, figsize=(3*N, 3*N),
        constrained_layout=True  # ✅ clean spacing
    )
    axs = axs.flatten()

    knots = layer.basis.knots.detach().cpu().numpy()

    for i, ax in enumerate(axs):
        if i < H:
            curve = B @ coeff[i].T
            ax.plot(x.squeeze(-1), curve, linewidth=1.4)

            # Put unit label inside
            ax.text(0.03, 0.95, f"Unit {i}",
                    fontsize=8, va="top", ha="left",
                    transform=ax.transAxes)

            # Knot lines
            for k in knots:
                ax.axvline(k, linestyle=":", color="gray", alpha=0.15)

            ax.set_xlim(0, 1)
            ax.grid(alpha=0.20, linestyle="--", linewidth=0.5)

            # ✅ Only show labels on leftmost + bottom row to avoid clutter
            row = i // N
            col = i % N
            if col != 0:
                ax.set_yticklabels([])
            if row != N - 1:
                ax.set_xticklabels([])
        else:
            ax.axis("off")

    # ✅ Shared axis labels without overlap
    fig.supxlabel("Input (Normalized)", fontsize=12)
    fig.supylabel("Spline Output", fontsize=12)

    fig.suptitle(
        f"Layer {layer_index} — Spline Functions for Feature {feature_index}",
        fontsize=15, fontweight="bold"
    )
    plt.show()
