import torch
from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty
from kolmarn.symbolic_regression import discover_symbolic_global, export_symbolic

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

    results = discover_symbolic_global(
        model,
        domain=(0.0, 1.0),
        n_samples=4096,
        maxsize=12,
        niterations=3000,
        timeout_s=120,
        unary_operators=["sin", "cos", "log", "exp"],
        binary_operators=["+", "-", "*", "/"],
    )

    print("\n=== Global Symbolic Regression Results ===")
    for r in results:
        print(f"R²={r.r2:.4f}  RMSE={r.rmse:.4f}  len={r.length}  f(x)≈ {export_symbolic(r,'string')}")
