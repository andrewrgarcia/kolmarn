"""
Hello, KolmArn!
---------------
A minimal example showing how to create, train, and visualize a
Kolmogorov-Arnold Network using the KolmArn package.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty
from kolmarn.visualize import plot_feature_splines

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    x_test = torch.linspace(0, 1, 200).unsqueeze(-1)
    y_true = torch.sin(2 * torch.pi * x_test)
    return (x, y), (x_test, y_true)

def build_model(in_features=1, layers=[32, 1], grid_size=16):
    return KANSequential(
        in_features=in_features,
        layer_sizes=layers,
        num_basis=grid_size,
        knots_trainable=False
    )

def train_model(model, data, lr=1e-2, epochs=1000, log_interval=200):
    x, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss += 1e-3 * spline_smoothness_penalty(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")

    return model

def visualize_results(model, data, test_data, visuals=True):
    x, y = data
    x_test, y_true = test_data
    with torch.no_grad():
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, alpha=0.3, label="data")
        plt.plot(x_test, model(x_test), label="KolmArn")
        plt.plot(x_test, y_true, label="True", linewidth=2, alpha=0.7)
        plt.legend()
        plt.title("KolmArn fit to sin(2πx)")
        plt.tight_layout()
        plt.show()

    if visuals: [plot_feature_splines(model, feature_index=0, layer_index=layer_id) for layer_id in [0,1]]

if __name__ == "__main__":
    (x, y), (x_test, y_true) = generate_data()
    model = build_model(layers=[32,1])
    trained_model = train_model(model, (x, y))
    visualize_results(trained_model, (x, y), (x_test, y_true))

