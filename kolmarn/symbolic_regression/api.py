# Post-training TRUE symbolic regression for KolmArn (PySR-based).
from __future__ import annotations

from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import numpy as np
from .types import SymbolicResult
from .utils import _fit_pysr, _get_kan_layer, _sample_unit_component
from .variables import _HAS_PYSR, _HAS_SYMPY
if _HAS_PYSR:
    from pysr import PySRRegressor
if _HAS_SYMPY:
    import sympy as sp


def export_symbolic(
    result: SymbolicResult,
    format: Literal["sympy", "latex", "torchcallable", "string"] = "sympy"
) -> Union["sp.Expr", str, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Export a discovered expression in various formats.
    """
    if format == "sympy":
        if not _HAS_SYMPY:
            raise RuntimeError("sympy is not available.")
        return result.expr_sympy
    elif format == "latex":
        return result.to_latex()
    elif format == "string":
        return result.expr_str or "(no expression)"
    elif format == "torchcallable":
        return result.to_callable_torch()
    else:
        raise ValueError(f"Unknown export format: {format}")


def run_pysr_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    maxsize: int = 12,
    niterations: int = 2000,
    timeout_s: Optional[int] = None,
    unary_operators: Optional[List[str]] = None,
    binary_operators: Optional[List[str]] = None,
    layer_index=None,
    out_index=None,
    in_index=None,
    n_points=None,
    domain=None,
) -> SymbolicResult:
    """Shared wrapper that runs PySR and returns a SymbolicResult."""
    expr_sympy, expr_str, metrics = _fit_pysr(
        X,
        y,
        maxsize=maxsize,
        niterations=niterations,
        timeout_in_seconds=timeout_s,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
    )

    return SymbolicResult(
        layer_index=layer_index,
        out_index=out_index,
        in_index=in_index,
        expr_sympy=expr_sympy,
        expr_str=expr_str,
        r2=float(metrics.get("r2", float("nan"))),
        rmse=float(metrics.get("rmse", float("nan"))),
        length=metrics.get("length"),
        n_points=n_points,
        x_domain=domain,
        notes=(metrics.get("notes") if isinstance(metrics.get("notes"), str) else None),
    )


def _discover_symbolic_batch(inputs, outputs, meta_fn=None, **kwargs):
    results = []
    for j in range(outputs.shape[1]):
        meta = meta_fn(j) if meta_fn else {}
        res = run_pysr_regression(
            inputs, outputs[:, j],
            **kwargs, **meta
        )
        results.append(res)
    return results


def discover_symbolic_form(
    model: torch.nn.Module,
    layer_index: int,
    out_index: int,
    in_index: int,
    *,
    n_points: int = 512,
    domain: Tuple[float, float] = (0.0, 1.0),
    component: Literal["spline_only", "with_linear"] = "spline_only",
    method: Literal["pysr"] = "pysr",
    maxsize: int = 12,
    niterations: int = 2000,
    timeout_s: Optional[int] = None,
    unary_operators: Optional[List[str]] = None,
    binary_operators: Optional[List[str]] = None,
) -> SymbolicResult:
    """
    Discover a compact, human-readable expression for a *single* univariate unit.

    Args mirror the roadmap (post-training TRUE SR). Defaults are lean & sane.
    """
    layer = _get_kan_layer(model, layer_index)
    x, y = _sample_unit_component(
        layer, out_index, in_index,
        n_points=n_points,
        domain=domain,
        component=component,
    )

    if method != "pysr":
        raise ValueError("Only 'pysr' is supported in this version (true SR).")

    return run_pysr_regression(
        x, y,
        maxsize=maxsize,
        niterations=niterations,
        timeout_s=timeout_s,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        layer_index=layer_index,
        out_index=out_index,
        in_index=in_index,
        n_points=n_points,
        domain=domain,
    )


def discover_symbolic_layer(
    model: torch.nn.Module,
    layer_index: int,
    *,
    n_points: int = 512,
    domain: Tuple[float, float] = (0.0, 1.0),
    component: Literal["spline_only", "with_linear"] = "spline_only",
    method: Literal["pysr"] = "pysr",
    maxsize: int = 12,
    niterations: int = 2000,
    timeout_s: Optional[int] = None,
    unary_operators: Optional[List[str]] = None,
    binary_operators: Optional[List[str]] = None,
    topk_by_coeff_norm: Optional[int] = None,
) -> List[SymbolicResult]:
    """
    Run SR over a whole layer. If topk_by_coeff_norm is set, only the most
    impactful (out,in) pairs by ||coeff[out,in,:]||_2 are processed.
    """
    layer = _get_kan_layer(model, layer_index)
    out_features, in_features, K = layer.coeff.shape

    pairs: List[Tuple[int, int]] = [(o, i) for o in range(out_features) for i in range(in_features)]

    if isinstance(topk_by_coeff_norm, int) and topk_by_coeff_norm > 0:
        # Rank by coefficient L2 norm
        norms = []
        with torch.no_grad():
            c = layer.coeff.detach().cpu()  # (out,in,K)
            for (o, i) in pairs:
                norms.append((float(torch.linalg.norm(c[o, i, :])), (o, i)))
        norms.sort(key=lambda t: t[0], reverse=True)
        pairs = [pair for _, pair in norms[:topk_by_coeff_norm]]

    results: List[SymbolicResult] = []
    for (o, i) in pairs:
        res = discover_symbolic_form(
            model, layer_index, o, i,
            n_points=n_points,
            domain=domain,
            component=component,
            method=method,
            maxsize=maxsize,
            niterations=niterations,
            timeout_s=timeout_s,
            unary_operators=unary_operators,
            binary_operators=binary_operators,
        )
        results.append(res)
    return results



@torch.no_grad()
def discover_symbolic_global(
    model: torch.nn.Module,
    *,
    X_domain=(0.0, 1.0),
    n_samples: int = 2048,
    device=None,
    maxsize: int = 12,
    niterations: int = 2000,
    timeout_s: int | None = None,
    unary_operators=None,
    binary_operators=None,
):
    """
    Stage-2 (Option A): run PySR symbolic regression on the *entire model output* f_model(X).
    Uses shared `run_pysr_regression` to generate SymbolicResult objects for each output dimension.
    """
    if not _HAS_PYSR:
        raise ImportError("PySR not installed. Install with `pip install pysr`.")

    model.eval()
    device = device or next(model.parameters()).device

    # Infer input dimension robustly
    try:
        in_features = getattr(model[0], "in_features", 1)
    except Exception:
        in_features = 1

    # Normalize domain shape
    if isinstance(X_domain[0], (int, float)):
        domain = [X_domain for _ in range(in_features)]
    else:
        domain = list(X_domain)

    # Generate random samples within domain
    X_np = np.zeros((n_samples, in_features), dtype=float)
    for j, (lo, hi) in enumerate(domain):
        X_np[:, j] = lo + (hi - lo) * np.random.rand(n_samples)
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)

    y_np = model(X).detach().cpu().numpy()
    if y_np.ndim == 1: y_np = y_np[:, None]

    return _discover_symbolic_batch(
        X_np, y_np,
        meta_fn=lambda j: {"out_index": j},
        maxsize=maxsize, niterations=niterations,
        timeout_s=timeout_s,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
    )