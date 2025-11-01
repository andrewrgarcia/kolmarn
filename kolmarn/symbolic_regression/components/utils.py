"""Low-level helpers for symbolic regression: PySR interface, sampling, layer access."""
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from .variables import _HAS_PYSR, _HAS_SYMPY
if _HAS_PYSR:
    from pysr import PySRRegressor
if _HAS_SYMPY:
    import sympy as sp



def _get_kan_layer(model: torch.nn.Module, layer_index: int):
    """Grab KAN layers by presence of 'coeff' attr (matches your code)."""
    kan_layers = [m for m in model.modules() if hasattr(m, "coeff")]
    if not kan_layers:
        raise ValueError("No KAN layers found in model.")
    if layer_index < 0 or layer_index >= len(kan_layers):
        raise IndexError(f"layer_index={layer_index} out of range [0, {len(kan_layers)-1}]")
    return kan_layers[layer_index]


def _sample_unit_component(
    layer,
    out_index: int,
    in_index: int,
    *,
    n_points: int = 512,
    domain: Tuple[float, float] = (0.0, 1.0),
    component: Literal["spline_only", "with_linear"] = "spline_only",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the univariate mapping for a single (out_index, in_index) pair.

    - spline_only:      f(x) = B(x) @ coeff[out,in,:]
    - with_linear:      f(x) = B(x) @ coeff[out,in,:] + W[out,in]*x + bias_share
                         where bias_share = bias[out] / in_features (simple, neutral split)

    Returns:
        x_np: shape (n_points, 1)
        y_np: shape (n_points, )
    """
    # Build grid
    xmin, xmax = domain
    x = torch.linspace(xmin, xmax, n_points).unsqueeze(1)  # (n, 1)

    # Evaluate basis on 1D input (feature slot)
    # layer.basis(x) -> (n, F, K). We feed as if it's a single-feature stream:
    B = layer.basis(x)[:, 0, :]  # (n, K)

    # Extract coeff vector for this out/in
    c = layer.coeff[out_index, in_index, :]  # (K,)
    spline_val = B @ c.T  # (n,)

    if component == "spline_only":
        y = spline_val
    else:
        # include linear skip contribution from this feature and a neutral bias share
        W = layer.linear.weight[out_index, in_index]  # scalar
        bias = layer.bias[out_index]                  # scalar
        in_features = layer.config.get("in_features", None)
        if in_features is None:
            # Fall back to estimating from coeff shape
            in_features = layer.coeff.shape[1]
        bias_share = bias / float(in_features)
        y = spline_val + W * x.squeeze(1) + bias_share

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    return x_np, y_np


def _fit_pysr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maxsize: int = 12,
    niterations: int = 2000,
    timeout_in_seconds: Optional[int] = None,
    unary_operators: Optional[List[str]] = None,
    binary_operators: Optional[List[str]] = None,
    extra_sympy_mappings: Optional[Dict[str, Callable]] = None,
    model_selection: Literal["best", "best_prune", "accuracy"] = "best",
) -> Tuple[Optional["sp.Expr"], Optional[str], Dict[str, Union[float, int]]]:
    """
    Run PySR on (x, y) with a minimal, robust configuration.
    Returns: (sympy_expr, expr_str, metrics_dict).
    """
    if not _HAS_PYSR:
        return None, None, {"r2": float("nan"), "rmse": float("nan"), "length": None, "notes": "PySR not installed."}
    if not _HAS_SYMPY:
        # Still can run PySR but can't return sympy expr; only string.
        pass

    # Default operator sets: compact and expressive
    if unary_operators is None:
        unary_operators = ["sin", "cos", "exp", "log"]
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]

    # Flatten x -> 1D array
    x1 = x.reshape(-1)
    X = x1[:, None]  # PySR expects (n_samples, n_features)

    reg = PySRRegressor(
        niterations=niterations,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        extra_sympy_mappings=extra_sympy_mappings,
        maxsize=maxsize,
        model_selection=model_selection,     # pick best by accuracy-simplicity tradeoff
        loss="L2DistLoss()",                 # MSE
        # Multi-objective: default includes complexity; Pareto across accuracy/complexity
        progress=False,
        timeout_in_seconds=timeout_in_seconds,
    )
    reg.fit(X, y)

    # Get best equation
    try:
        expr_str = reg.get_best()["equation"]  # string
    except Exception:
        expr_str = None

    expr_sympy = None
    if _HAS_SYMPY and hasattr(reg, "sympy") and expr_str is not None:
        try:
            expr_sympy = reg.sympy()
        except Exception:
            expr_sympy = None

    # Metrics (use available predict + length)
    try:
        yhat = reg.predict(X).reshape(-1)
        resid = yhat - y.reshape(-1)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        # Safe R^2
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        ss_res = float(np.sum((y - yhat) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("inf"))
    except Exception:
        rmse, r2 = float("nan"), float("nan")

    # Expression length/complexity (best model length if available)
    try:
        length = int(reg.get_best()["length"])
    except Exception:
        length = None

    return expr_sympy, expr_str, {"r2": r2, "rmse": rmse, "length": length, "notes": None}

