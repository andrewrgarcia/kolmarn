# Kolmarn 

Clean and interpretable implementation of Kolmogorov-Arnold Networks (KANs).

**K**olm**A**r**n** provides smooth univariate spline layers with visualization and regularization tools designed for scientific machine learning, forecasting, and high-stakes decision applications.

---

## Motivation

Many neural networks achieve accuracy at the expense of transparency.  
KolmArn enables explicit functional decomposition:

> Any multivariate function can be represented as sums of univariate nonlinearities.

KolmArn implements this principle to deliver:
- Explicit learned functions per input feature
- Smooth basis expansions for stability
- Visualization utilities for model interpretation
- Compact architectures effective with limited data

This makes KolmArn well-suited for:
- Econometrics and macroeconomic forecasting
- Financial modeling and risk systems
- Scientific ML and physical systems
- Settings requiring model accountability

---

## Installation

Local development:

```bash
git clone https://github.com/YOURNAME/kolmarn.git
cd kolmarn
pip install -e .
```

Requires: Python ≥ 3.9, PyTorch ≥ 2.0

---

## Features

| Component                                  |    Status   |
| ------------------------------------------ | :---------: |
| KAN layers with spline expansions          |      ✔      |
| Basis options: KAN spline and RBF          |      ✔      |
| Smoothness regularization                  |      ✔      |
| Visualization of learned feature functions |      ✔      |
| Input normalization utilities              | In Progress |
| Trainable knot positions                   | In Progress |
| Symbolic function export                   |   Planned   |
| Real-world forecasting examples            |   Planned   |

---

## Example Usage

```python
import torch
from kolmarn.models import KANSequential

# Learn sin(2πx)
x = torch.rand(200, 1)
y = torch.sin(2 * torch.pi * x)

model = KANSequential(
    in_features=1,
    layer_sizes=[32, 1],
    num_basis=16
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Visualize learned splines for input 0
from kolmarn.visualize import plot_feature_splines
plot_feature_splines(model, feature_index=0)
```

---

## Interpretability

KolmArn exposes the learned nonlinear transformations applied to each input feature.
These can be plotted directly to understand where and how the model responds to changes in the data.

Example output: learned univariate spline functions per hidden unit in a layer.

(Visual examples to be included with documentation.)

---

## Performance Demonstration

Repeated regression experiments on `sin(2πx)` (100 independent trials, equal parameter counts):

| Metric | KolmArn (mean ± std) | MLP (mean ± std) |
|-------|---------------------:|----------------:|
| RMSE  | 0.0167 ± 0.0079       | 0.0185 ± 0.0095 |
| MAE   | 0.0133 ± 0.0079       | 0.0148 ± 0.0078 |
| R²    | 0.9993 ± 0.0012       | 0.9991 ± 0.0009 |

KolmArn matches or exceeds MLP accuracy while exposing its learned structure for direct interpretation.

---

## Roadmap

Planned enhancements include:

* Automatic input scaling
* Trainable knot adaptation
* Visualization of knot movement
* Symbolic extraction of spline equations
* Domain-specific examples (econometrics, risk models)

---

## License

MIT — open for academic and commercial use.

