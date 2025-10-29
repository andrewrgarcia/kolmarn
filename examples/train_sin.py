import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from kan.regularizers import spline_smoothness_penalty

def data_generation():
    N = 200
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)

    x_test = torch.linspace(0, 1, 200).unsqueeze(-1)
    y_true = torch.sin(2 * torch.pi * x_test)
    return (x, y), (x_test, y_true)


def train_models(data):
    x, y = data
    opt_kan = torch.optim.Adam(kan.parameters(), lr=1e-2)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-2)

    for i in range(1000):
        # KAN
        kan_pred = kan(x)
        loss_kan = ((kan_pred - y) ** 2).mean()
        loss_kan += 1e-3 * spline_smoothness_penalty(kan)  # smoothness
        opt_kan.zero_grad()
        loss_kan.backward()
        opt_kan.step()

        # MLP
        mlp_pred = mlp(x)
        loss_mlp = ((mlp_pred - y) ** 2).mean()
        opt_mlp.zero_grad()
        loss_mlp.backward()
        opt_mlp.step()

def visualize_plot(data, test_data):
    x, y = data
    x_test, y_true = test_data

    plt.figure(figsize=(8, 4))
    with torch.no_grad():
        plt.scatter(x, y, alpha=0.3, label="data")
        plt.plot(x_test, mlp(x_test), label="MLP", linestyle="--")
        plt.plot(x_test, kan(x_test), label="KAN")
        plt.plot(x_test, y_true, label="True", linewidth=2, alpha=0.7)
    plt.legend()
    plt.title("KAN vs MLP on sin(2πx)")
    plt.show()



def performance_metrics(test_data):
    x_test, y_true = test_data

    with torch.no_grad():
        y_kan = kan(x_test).numpy()
        y_mlp = mlp(x_test).numpy()
        y_true_np = y_true.numpy()

    rmse_kan = np.sqrt(mean_squared_error(y_true_np, y_kan))
    rmse_mlp = np.sqrt(mean_squared_error(y_true_np, y_mlp))

    r2_kan = r2_score(y_true_np, y_kan)
    r2_mlp = r2_score(y_true_np, y_mlp)

    mae_kan = mean_absolute_error(y_true_np, y_kan)
    mae_mlp = mean_absolute_error(y_true_np, y_mlp)

    print("\nPerformance Metrics:")
    print(f"{'Model':>10} | {'RMSE':>8} | {'MAE':>8} | {'R^2':>8}")
    print("-"*40)
    print(f"{'KAN':>10} | {rmse_kan:8.4f} | {mae_kan:8.4f} | {r2_kan:8.4f}")
    print(f"{'MLP':>10} | {rmse_mlp:8.4f} | {mae_mlp:8.4f} | {r2_mlp:8.4f}")


if __name__ == "__main__":
    from kan.models import KANSequential

    kan = KANSequential(1, [32, 1], num_basis=16)
    mlp = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 1)
    )

    data, test_data = data_generation()
    train_models(data)
    visualize_plot(data, test_data)
    performance_metrics(test_data)