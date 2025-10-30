import torch
import matplotlib.pyplot as plt
from kolmarn.models import KANSequential
from kolmarn.regularizers import spline_smoothness_penalty, knot_spacing_penalty
from kolmarn.visualize import plot_feature_splines, plot_knots_evolution

def generate_data(N=200):
    x = torch.rand(N, 1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    x_test = torch.linspace(0, 1, 200).unsqueeze(-1)
    y_true = torch.sin(2 * torch.pi * x_test)
    return (x, y), (x_test, y_true)


def train_model(model, data, lr=1e-2, epochs=1000, log_interval=200, visuals=True):
    x, y = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss += 1e-3 * spline_smoothness_penalty(model)
        loss += 1e-3 * knot_spacing_penalty(model)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")
            if visuals: plot_knots_evolution(model, epoch, show=True)

    return model


def visualize_results(model, data, test_data):
    x, y = data
    x_test, y_true = test_data
    with torch.no_grad():
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, alpha=0.3, label="data")
        plt.plot(x_test, model(x_test), label="KAN (adaptive)")
        plt.plot(x_test, y_true, label="True", linewidth=2, alpha=0.7)
        plt.legend()
        plt.title("Adaptive KAN on sin(2πx)")
        plt.tight_layout()
        plt.show()

    plot_feature_splines(model, feature_index=0)


if __name__ == "__main__":
    (x, y), (x_test, y_true) = generate_data()

    model = KANSequential(
        in_features=1,
        layer_sizes=[32, 1],
        num_basis=16,
        knots_trainable=False,
    )

    # Enable adaptive knots in each layer
    for layer in model.layers:
        layer.basis.enable_adaptive_knots()


    trained_model = train_model(model, (x, y), visuals=False)
    visualize_results(trained_model, (x, y), (x_test, y_true))
