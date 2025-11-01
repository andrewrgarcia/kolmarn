<p align="center">
  <img width="300" alt="kolmarn" src="https://github.com/user-attachments/assets/566667a1-4da5-4437-b60a-2af7086f9db8" />
</p>
<h1 align="center">KOLMARN</h1>
<p align="center">
  Clean and interpretable implementation of Kolmogorov-Arnold Networks (KANs)
</p>
<p align="center"><em>Kolmogorov–Arnold Networks, reimagined for clarity.</em></p>

<p align="center">
  <a href="https://pypi.org/project/kolmarn/"><img src="https://img.shields.io/pypi/v/kolmarn?color=blue&label=PyPI&style=flat-square"></a>
  <a href="https://github.com/andrewrgarcia/kolmarn/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=flat-square"></a>
</p>

---

**K**olm**a**r**n** provides smooth univariate spline layers with visualization and regularization tools designed for scientific machine learning, forecasting, and high-stakes decision applications.

---

## Motivation

Many neural networks achieve accuracy at the expense of transparency.  
Kolmarn enables explicit functional decomposition:

> Any multivariate function can be represented as sums of univariate nonlinearities.

Kolmarn builds on the formulation introduced in *KAN: Kolmogorov–Arnold Networks* (Liu et al., 2024) [[arXiv:2404.19756](https://arxiv.org/abs/2404.19756)], providing a simplified and interpretable PyTorch implementation for research and education.

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

### From PyPI

```bash
pip install kolmarn
```

### Local development (for latest dev version)

```bash
git clone https://github.com/andrewrgarcia/kolmarn.git
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

A minimal end-to-end example that trains and visualizes a Kolmogorov–Arnold Network on `sin(2πx)` is available at:

[`examples/hello_kolmarn.py`](./examples/hello_kolmarn.py)

```bash
python examples/hello_kolmarn.py
```

This script:

* Generates synthetic data
* Trains a small KAN to approximate `sin(2πx)`
* Applies smoothness regularization
* Visualizes both the fit and layer-wise spline functions

---

## Interpretability

Kolmarn exposes the learned nonlinear transformations applied to each input feature.
These can be plotted directly to understand where and how the model responds to changes in the data.

Example output: learned univariate spline functions per hidden unit in a layer.
(Visual examples will be included with documentation.)

---

## Performance Demonstration

Repeated regression experiments on `sin(2πx)` (100 independent trials, equal parameter counts):

| Metric | KAN (mean ± std) | MLP (mean ± std) |
| :----- | ---------------: | ---------------: |
| RMSE   |  0.0167 ± 0.0079 |  0.0185 ± 0.0095 |
| MAE    |  0.0133 ± 0.0079 |  0.0148 ± 0.0078 |
| R_sq   |  0.9993 ± 0.0012 |  0.9991 ± 0.0009 |

Kolmarn matches or exceeds MLP accuracy while exposing its learned structure for direct interpretation.

---

## Comparative Evaluation

To assess implementation performance and numerical stability, we benchmarked **Kolmarn** against the official [PyKAN](https://github.com/KindXiaoming/pykan) implementation under identical conditions:

* Function: `sin(2πx)`
* Equal parameter counts (`width=[1, 32, 1]`, grid = 16)
* Optimizer: Adam, 1000 training steps
* 100 independent trials with randomized initialization

| Model                                                                |        RMSE ↓       |         R_sq ↑        |
| :------------------------------------------------------------------- | :-----------------: | :-----------------: |
| **Kolmarn (ours)**                                                   | **0.0175 ± 0.0000** | **0.9994 ± 0.0000** |
| PyKAN ([KindXiaoming et al.](https://github.com/KindXiaoming/pykan)) |   0.0300 ± 0.0001   |   0.9982 ± 0.0000   |

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

## Citation

If you use **Kolmarn** in academic work, please cite it as:

```bibtex
@software{kolmarn,
  author = {Garcia, Andrew R.},
  title = {Kolmarn: Clean and Interpretable Kolmogorov–Arnold Networks},
  year = {2025},
  url = {https://github.com/andrewrgarcia/kolmarn}
}
```

Kolmarn builds on the original formulation presented in:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

---

## License

MIT — open for academic and commercial use.

