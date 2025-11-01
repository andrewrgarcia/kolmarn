from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from .components import SymbolicResult, _fit_pysr
from copy import deepcopy
import time


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
    start_t = time.perf_counter()
    expr_sympy, expr_str, metrics = _fit_pysr(
        X,
        y,
        maxsize=maxsize,
        niterations=niterations,
        timeout_in_seconds=timeout_s,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
    )
    end_t = time.perf_counter()

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
        runtime_s=end_t - start_t,            # total seconds
        start_time=start_t,                   # optional start mark
    )


def _discover_symbolic_batch(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    meta_fn: Optional[Callable[[int], dict]] = None,
    tol: float = 1e-3,
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
        if sr_kwargs.get("ensemble_runs", 1) > 1:
            res = discover_symbolic_ensemble(
                inputs, outputs[:, j],
                n_runs=sr_kwargs["ensemble_runs"],
                perturb=sr_kwargs.get("ensemble_perturb", 0.01),
                tol=sr_kwargs.get("tolerance", 1e-3),
                **{k: v for k, v in sr_kwargs.items() if k not in ["ensemble_runs", "ensemble_perturb", "tolerance"]},
                **meta,
            )
        else:
            res = run_pysr_regression(inputs, outputs[:, j], **sr_kwargs, **meta)
            res = postprocess_symbolic(res, inputs, outputs[:, j], sr_kwargs.get("tolerance", 1e-3))
        results.append(res)
    return results


def discover_symbolic_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_runs: int = 5,
    perturb: float = 0.01,
    tol: float = 1e-3,
    **sr_kwargs,
) -> SymbolicResult:
    """
    Run multiple PySR fits under small perturbations and summarize stability.
    Returns a SymbolicResult with ensemble-level metrics.
    """

    results = []
    for i in range(n_runs):
        Xp = X + np.random.normal(0, perturb, X.shape)
        res = run_pysr_regression(Xp, y, **sr_kwargs)
        res = postprocess_symbolic(res, Xp, y, tol)
        results.append(res)

    # Choose the best by R_sq, then attach stability summary
    best = max(results, key=lambda r: r.r2 if not np.isnan(r.r2) else -np.inf)
    best = deepcopy(best)
    best = best.summarize_ensemble(results)
    return best


def postprocess_symbolic(result: SymbolicResult,
                         X: np.ndarray,
                         y: np.ndarray,
                         tol: float = 1e-3) -> SymbolicResult:
    """
    Estimate residual noise and prune small symbolic coefficients.
    """
    result.estimate_noise(X, y)
    result.prune(tol)
    return result
