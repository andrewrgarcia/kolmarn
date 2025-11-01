from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .components import SymbolicResult, _fit_pysr


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


def _discover_symbolic_batch(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    meta_fn: Optional[Callable[[int], dict]] = None,
    **sr_kwargs,
) -> List[SymbolicResult]:
    """
    Internal helper to run symbolic regression across multiple output columns.

    Args:
        inputs: X matrix (n_samples, n_features)
        outputs: Y matrix (n_samples, n_targets)
        meta_fn: function j -> metadata kwargs (e.g. out_index)
        sr_kwargs: forwarded args to run_pysr_regression()
    """
    results = []
    n_targets = outputs.shape[1]
    for j in range(n_targets):
        meta = meta_fn(j) if meta_fn else {}
        res = run_pysr_regression(inputs, outputs[:, j], **sr_kwargs, **meta)
        results.append(res)
    return results
