import numpy as np
from .train_sin import (
    data_generation,
    train_models,
)
from kolmarn.models import KANSequential
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import wilcoxon


def evaluate_model_once():
    """Train fresh KAN+MLP models once and return metrics."""

    # create new fresh models per trial
    global kan, mlp
    kan = KANSequential(1, [32, 1], num_basis=16)
    mlp = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 1)
    )

    data, test_data = data_generation()
    train_models(kan, mlp, data)

    x_test, y_true = test_data
    with torch.no_grad():
        y_kan = kan(x_test).numpy()
        y_mlp = mlp(x_test).numpy()
        y_true_np = y_true.numpy()

    rmse_kan = np.sqrt(mean_squared_error(y_true_np, y_kan))
    rmse_mlp = np.sqrt(mean_squared_error(y_true_np, y_mlp))

    mae_kan = mean_absolute_error(y_true_np, y_kan)
    mae_mlp = mean_absolute_error(y_true_np, y_mlp)

    r2_kan = r2_score(y_true_np, y_kan)
    r2_mlp = r2_score(y_true_np, y_mlp)

    return (rmse_kan, mae_kan, r2_kan), (rmse_mlp, mae_mlp, r2_mlp)


def run_repeated_trials(num_trials=20):
    """Benchmark metrics across multiple training trials."""

    rmse_kan_list, rmse_mlp_list = [], []
    mae_kan_list, mae_mlp_list = [], []
    r2_kan_list, r2_mlp_list = [], []

    for i in range(num_trials):
        (rmse_kan, mae_kan, r2_kan), (rmse_mlp, mae_mlp, r2_mlp) = evaluate_model_once()

        rmse_kan_list.append(rmse_kan)
        rmse_mlp_list.append(rmse_mlp)
        mae_kan_list.append(mae_kan)
        mae_mlp_list.append(mae_mlp)
        r2_kan_list.append(r2_kan)
        r2_mlp_list.append(r2_mlp)

        print(f"Trial {i+1:02d} | "
              f"KAN RMSE={rmse_kan:.4f}  MLP RMSE={rmse_mlp:.4f}")

    print("\n==== Summary Across Trials ====")
    print(f"Trials: {num_trials}\n")

    def fmt(a): return f"{np.mean(a):.4f} ± {np.std(a):.4f}"

    print(f"{'Metric':<15} {'KAN':>20} {'MLP':>20}")
    print("-" * 60)
    print(f"{'RMSE':<15} {fmt(rmse_kan_list):>20} {fmt(rmse_mlp_list):>20}")
    print(f"{'MAE':<15} {fmt(mae_kan_list):>20} {fmt(mae_mlp_list):>20}")
    print(f"{'R_sq':<15} {fmt(r2_kan_list):>20} {fmt(r2_mlp_list):>20}")

    # ---- Significance Testing (paired, non-parametric) ----
    stat_rmse, p_rmse = wilcoxon(rmse_kan_list, rmse_mlp_list, alternative="less")
    stat_mae, p_mae = wilcoxon(mae_kan_list, mae_mlp_list, alternative="less")
    stat_r2,  p_r2  = wilcoxon(r2_kan_list,  r2_mlp_list,  alternative="greater")

    print("\nStatistical Significance (Wilcoxon Signed-Rank Test):")
    print(f"{'Metric':<10} {'p-value':>12} {'Significant?':>15}")
    print("-" * 40)
    print(f"{'RMSE':<10} {p_rmse:12.4e} {('YES' if p_rmse < 0.05 else 'NO'):>15}")
    print(f"{'MAE':<10} {p_mae:12.4e} {('YES' if p_mae < 0.05 else 'NO'):>15}")
    print(f"{'R_sq':<10} {p_r2:12.4e} {('YES' if p_r2 < 0.05 else 'NO'):>15}")


if __name__ == "__main__":
    run_repeated_trials(num_trials=100)
