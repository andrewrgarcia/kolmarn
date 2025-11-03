import torch
from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty
from kolmarn.symbolic_regression import discover_symbolic_global, export_symbolic
from kolmarn.symbolic_regression.components import GlobalSymbolicConfig

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    return x, y

def train(model, x, y, steps=800, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss += 1e-3 * spline_smoothness_penalty(model)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

if __name__ == "__main__":
    x, y = generate_data()
    model = KANSequential(1, [32, 1], num_basis=16, knots_trainable=False)
    model = train(model, x, y)

    cfg = GlobalSymbolicConfig(
        domain=(0.0, 1.0),
        unary_operators=["sin", "cos", "log", "exp"],
        binary_operators=["+", "-", "*", "/"],
        mode="fast",   # ⚡ switch to "full" for final publication runs
    )

    results = discover_symbolic_global(model, config=cfg)

    print("\n=== Global Symbolic Regression Results ===")
    if results:
        total_time = getattr(results[0], "total_runtime_s", None)
        for r in results:
            print(
                f"R_sq={r.r2:.4f}  RMSE={r.rmse:.4f}  "
                f"std_dev={(r.noise_std or float('nan')):.4f}  "
                f"stability={(r.expr_stability or float('nan')):.3f}  "
                f"len={r.length}  time={r.runtime_s:.1f}s"
            )
            print("   →", export_symbolic(r, "string"))

        if total_time is not None:
            print(f"\nTotal runtime across outputs: {total_time:.1f}s")
