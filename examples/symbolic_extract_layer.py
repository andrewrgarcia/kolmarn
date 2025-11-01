"""
Example: Discover symbolic equations from a trained KolmArn layer.
Requires extras: pip install -e ".[symreg]"
"""

import torch
from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty
from kolmarn.symbolic_regression import discover_symbolic_layer, export_symbolic
from kolmarn.symbolic_regression.components import SymbolicConfig

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    return x, y

def train(model, x, y, steps=800, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss += 1e-3 * spline_smoothness_penalty(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

if __name__ == "__main__":
    x, y = generate_data()
    model = KANSequential(1, [32, 1], num_basis=16, knots_trainable=False)
    model = train(model, x, y)

    # Discover equations for top-6 most influential (out,in) pairs in first layer
    cfg = SymbolicConfig(
        n_points=512,
        component="spline_only",
        maxsize=10,
        niterations=1500,
        timeout_s=60,
        topk_by_coeff_norm=6,
    )

    try:
        results = discover_symbolic_layer(model, layer_index=0, config=cfg)
    except RuntimeError as e:
        print("WARNING: Symbolic regression backend unavailable:", e)
        exit(0)

    print("\n=== Symbolic Regression Results (Layer 0) ===")
    for r in results:
        eq = export_symbolic(r, "string")
        print(
            f"[L{r.layer_index} o{r.out_index} <- i{r.in_index}] "
            f"R_sq={r.r2:.4f} RMSE={r.rmse:.4f} len={r.length or '-'}  f(x)≈ {eq}"
        )

    # Example: convert first expression to callable and evaluate
    if results and results[0].expr_sympy is not None:
        f = export_symbolic(results[0], "torchcallable")
        with torch.no_grad():
            xt = torch.linspace(0, 1, 10).unsqueeze(1)
            print("\nSample callable output:", f(xt).squeeze(-1)[:5])
