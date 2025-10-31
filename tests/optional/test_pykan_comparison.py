import torch
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from kolmarn.models import KANSequential

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    x_test = torch.linspace(0, 1, 200).unsqueeze(-1)
    y_true = torch.sin(2 * torch.pi * x_test)
    return (x, y), (x_test, y_true)


def train_model(model, x, y, steps=1000, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def run_single_trial():
    try:
        from kan import KAN as PyKAN
    except ImportError:
        print("WARNING: PyKAN not installed; skipping comparison test.")
        return None

    (x, y), (x_test, y_true) = generate_data()

    # --- KolmArn model ---
    kan = KANSequential(1, [32, 1], num_basis=16)
    kan = train_model(kan, x, y)

    # --- PyKAN model ---
    pykan_model = PyKAN(width=[1, 32, 1], grid=16, k=3)

    # Try legacy vs. new API
    try:
        # Legacy API (custom .train with args)
        pykan_model.train({"train_input": x, "train_label": y}, opt="Adam", steps=1000, lr=1e-2)
    except TypeError:
        # Newer API (manual training)
        opt = torch.optim.Adam(pykan_model.parameters(), lr=1e-2)
        for _ in range(1000):
            pred = pykan_model(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # --- Evaluation ---
    with torch.no_grad():
        y_kan = kan(x_test).numpy()
        y_pykan = pykan_model(x_test).numpy()
        y_true_np = y_true.numpy()

    rmse_k, r2_k = evaluate(y_true_np, y_kan)
    rmse_p, r2_p = evaluate(y_true_np, y_pykan)

    return rmse_k, r2_k, rmse_p, r2_p


@pytest.mark.optional
def test_compare_kolmarn_vs_pykan(num_trials=10):
    results = [run_single_trial() for _ in range(num_trials)]
    results = np.array(results)

    rmse_k, r2_k, rmse_p, r2_p = results[:, 0], results[:, 1], results[:, 2], results[:, 3]

    print("\n==== KolmArn vs PyKAN (sin(2πx)) ====")
    print(f"Trials: {num_trials}")
    print(f"{'Metric':<8} | {'KolmArn Mean ± Std':>25} | {'PyKAN Mean ± Std':>25}")
    print("-" * 65)
    print(f"RMSE    | {rmse_k.mean():.4f} ± {rmse_k.std():.4f} | {rmse_p.mean():.4f} ± {rmse_p.std():.4f}")
    print(f"R²      | {r2_k.mean():.4f} ± {r2_k.std():.4f} | {r2_p.mean():.4f} ± {r2_p.std():.4f}")

    assert r2_k.mean() > 0.99 and r2_p.mean() > 0.99


if __name__ == "__main__":
    num_trials = 100
    results = [run_single_trial() for _ in range(num_trials)]
    results = np.array(results)

    rmse_k, r2_k, rmse_p, r2_p = results[:, 0], results[:, 1], results[:, 2], results[:, 3]

    print("\n==== KolmArn vs PyKAN (Standalone Run) ====")
    print(f"Trials: {num_trials}")
    print(f"{'Metric':<8} | {'KolmArn Mean ± Std':>25} | {'PyKAN Mean ± Std':>25}")
    print("-" * 65)
    print(f"RMSE    | {rmse_k.mean():.4f} ± {rmse_k.std():.4f} | {rmse_p.mean():.4f} ± {rmse_p.std():.4f}")
    print(f"R²      | {r2_k.mean():.4f} ± {r2_k.std():.4f} | {r2_p.mean():.4f} ± {r2_p.std():.4f}")
