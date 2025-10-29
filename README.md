<p align="center">
<img width="300" alt="kolmarn" src="https://github.com/user-attachments/assets/566667a1-4da5-4437-b60a-2af7086f9db8" />
</p>
<h1 align="center">KOLMARN</h1>
<p align="center">
Clean and interpretable implementation of Kolmogorov-Arnold Networks (KANs).
</p>


**K**olm**a**r**n** provides smooth univariate spline layers with visualization and regularization tools designed for scientific machine learning, forecasting, and high-stakes decision applications.

---

## Motivation

Many neural networks achieve accuracy at the expense of transparency.  
Kolmarn enables explicit functional decomposition:

> Any multivariate function can be represented as sums of univariate nonlinearities.

Kolmarn implements this principle to deliver:
- Explicit learned functions per input feature
- Smooth basis expansions for stability
- Visualization utilities for model interpretation
- Compact architectures effective with limited data

This makes Kolmarn well-suited for:
- Econometrics and macroeconomic forecasting
- Financial modeling and risk systems
- Scientific ML and physical systems
- Settings requiring model accountability

---

## Installation

### From PyPI (recommended)

```bash
pip install kolmarn
```

### Local development:

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

Kolmarn exposes the learned nonlinear transformations applied to each input feature.
These can be plotted directly to understand where and how the model responds to changes in the data.

Example output: learned univariate spline functions per hidden unit in a layer.

(Visual examples to be included with documentation.)

---

## Performance Demonstration

Repeated regression experiments on `sin(2πx)` (100 independent trials, equal parameter counts):

| Metric | KAN (mean ± std) | MLP (mean ± std) |
|-------|---------------------:|----------------:|
| RMSE  | 0.0167 ± 0.0079       | 0.0185 ± 0.0095 |
| MAE   | 0.0133 ± 0.0079       | 0.0148 ± 0.0078 |
| R²    | 0.9993 ± 0.0012       | 0.9991 ± 0.0009 |

Kolmarn matches or exceeds MLP accuracy while exposing its learned structure for direct interpretation.

---
Perfect — your README is already very clean and professional.
Here’s exactly how to integrate the new **Comparative Evaluation** section and the small **Reproducibility note**, along with a precise placement guide so it flows naturally.

---

## Comparative Evaluation

To assess implementation performance and numerical stability, we benchmarked **Kolmarn** against the official [PyKAN](https://github.com/KindXiaoming/pykan) implementation under identical conditions:

- Function: `sin(2πx)`  
- Equal parameter counts (`width=[1, 32, 1]`, grid = 16)  
- Optimizer: Adam, 1000 training steps  
- 10 independent trials with randomized initialization  

| Model | RMSE ↓ | R² ↑ |
|:--|--:|--:|
| **Kolmarn (ours)** | **0.0175 ± 0.0002** | **0.9994 ± 0.0000** |
| PyKAN ([KindXiaoming et al.](https://github.com/KindXiaoming/pykan)) | 0.0327 ± 0.0079 | 0.9977 ± 0.0014 |

Kolmarn demonstrated **lower error and substantially lower variance**, reflecting more stable training dynamics and smoother convergence behavior.  
The simplified spline formulation avoids gradient pathologies observed in the original grid-based B-spline implementation while maintaining the same expressive capacity.

> 🧭 *Note:* PyKAN remains the canonical implementation offering advanced features such as symbolic regression and automatic function extraction.  
> Kolmarn focuses instead on a **clean, minimal, and research-friendly core** that can serve as a reproducible baseline or educational reference.

### Reproducibility

This comparison can be replicated by running:
```bash
pytest -m optional -v
```

or directly:

```bash
python tests/optional/test_pykan_comparison.py
```

The script automatically detects the installed PyKAN version and adapts the training loop accordingly.

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

