import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty, knot_spacing_penalty
from kolmarn.visualize import plot_feature_splines, plot_knots_evolution

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    x_test = torch.linspace(0, 1, 200).unsqueeze(-1)
    y_true = torch.sin(2 * torch.pi * x_test)
    return (x, y), (x_test, y_true)


def train_model(model, data, lr=1e-2, epochs=800, adaptive=False, log_interval=200, visuals=False):
    x, y = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss += 1e-3 * spline_smoothness_penalty(model)
        if adaptive:
            loss += 1e-3 * knot_spacing_penalty(model)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")
            if visuals and adaptive: plot_knots_evolution(model, epoch, show=True)
    return model


def evaluate(model, x_test, y_true):
    with torch.no_grad():
        y_pred = model(x_test).cpu().numpy()
        y_true = y_true.cpu().numpy()
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


def visualize(models, names, data, test_data):
    x, y = data
    x_test, y_true = test_data
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, alpha=0.3, label="data")

    with torch.no_grad():
        for name, model in zip(names, models):
            plt.plot(x_test, model(x_test), label=name)
        plt.plot(x_test, y_true, label="True", linewidth=2, alpha=0.7)

    plt.legend()
    plt.title("KAN (adaptive vs static) and MLP on sin(2πx)")
    plt.tight_layout()
    plt.show()

    for i, (model, name) in enumerate(zip(models, names)):
        if isinstance(model, KANSequential):
            print(f"\nSpline visualization — {name}")
            plot_feature_splines(model, feature_index=0)


if __name__ == "__main__":
    (x, y), (x_test, y_true) = generate_data()

    kan_adapt = KANSequential(1, [32, 1], num_basis=16, knots_trainable=False)
    for layer in kan_adapt.layers:
        layer.basis.enable_adaptive_knots()

    kan_static = KANSequential(1, [32, 1], num_basis=16, knots_trainable=False)

    mlp = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 1)
    )

    print("Training adaptive KAN...")
    kan_adapt = train_model(kan_adapt, (x, y), adaptive=True)

    print("Training static KAN...")
    kan_static = train_model(kan_static, (x, y), adaptive=False)

    print("Training MLP...")
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(1000):
        pred = mlp(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    models = [kan_adapt, kan_static, mlp]
    names = ["KAN (adaptive)", "KAN (static)", "MLP"]
    results = {name: evaluate(model, x_test, y_true) for name, model in zip(names, models)}

    print("\n===Performance Comparison===")
    print(f"{'Model':<16} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8}")
    print("-"*45)
    for name in names:
        m = results[name]
        print(f"{name:<16} | {m['rmse']:8.4f} | {m['mae']:8.4f} | {m['r2']:8.4f}")


    visualize(models, names, (x, y), (x_test, y_true))
