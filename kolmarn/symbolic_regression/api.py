# Post-training TRUE symbolic regression for KolmArn (PySR-based).
from __future__ import annotations

from typing import Callable, List, Literal, Optional, Tuple, Union

import torch

from .types import SymbolicResult
from .utils import _fit_pysr, _get_kan_layer, _sample_unit_component
from .variables import _HAS_SYMPY
if _HAS_SYMPY:
    import sympy as sp
    
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

    expr_sympy, expr_str, metrics = _fit_pysr(
        x, y,
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
        n_points=int(n_points),
        x_domain=domain,
        notes=(metrics.get("notes") if isinstance(metrics.get("notes"), str) else None),
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
