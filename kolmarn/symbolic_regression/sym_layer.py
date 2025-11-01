from __future__ import annotations

from typing import Callable, List, Literal, Optional, Tuple, Union

import torch

from .helpers import run_pysr_regression
from .components import SymbolicResult, SymbolicConfig, _get_kan_layer, _sample_unit_component, _HAS_PYSR, _HAS_SYMPY
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


def discover_symbolic_form(
    model: torch.nn.Module,
    layer_index: int,
    out_index: int,
    in_index: int,
    *,
    config: SymbolicConfig = SymbolicConfig(),
) -> SymbolicResult:
    """
    Discover a compact, human-readable expression for a *single* univariate unit.

    Args mirror the roadmap (post-training TRUE SR). Defaults are lean & sane.
    """
    layer = _get_kan_layer(model, layer_index)
    x, y = _sample_unit_component(
        layer, out_index, in_index,
        n_points=config.n_points,
        domain=config.domain,
        component=config.component,
    )

    if config.method != "pysr":
        raise ValueError("Only 'pysr' is supported in this version (true SR).")

    return run_pysr_regression(
        x, y,
        maxsize=config.maxsize,
        niterations=config.niterations,
        timeout_s=config.timeout_s,
        unary_operators=config.unary_operators,
        binary_operators=config.binary_operators,
        layer_index=layer_index,
        out_index=out_index,
        in_index=in_index,
        n_points=config.n_points,
        domain=config.domain,
    )


def discover_symbolic_layer(
    model: torch.nn.Module,
    layer_index: int,
    *,
    component: Literal["spline_only", "with_linear"] = "spline_only",
    topk_by_coeff_norm: Optional[int] = None,
    config: SymbolicConfig = SymbolicConfig(),
) -> List[SymbolicResult]:
    """
    Run SR over a whole layer. If topk_by_coeff_norm is set, only the most
    impactful (out,in) pairs by ||coeff[out,in,:]||_2 are processed.
    """
    layer = _get_kan_layer(model, layer_index)
    out_features, in_features, K = layer.coeff.shape

    pairs = [(o, i) for o in range(out_features) for i in range(in_features)]
    if isinstance(topk_by_coeff_norm, int) and topk_by_coeff_norm > 0:
        with torch.no_grad():
            c = layer.coeff.detach().cpu()
            norms = [(float(torch.linalg.norm(c[o, i, :])), (o, i)) for (o, i) in pairs]
        norms.sort(key=lambda t: t[0], reverse=True)
        pairs = [pair for _, pair in norms[:topk_by_coeff_norm]]

    results = []
    for (o, i) in pairs:
        x, y = _sample_unit_component(
            layer, o, i,
            n_points=config.n_points,
            domain=config.domain,
            component=component,
        )
        res = run_pysr_regression(
            x, y,
            maxsize=config.maxsize,
            niterations=config.niterations,
            timeout_s=config.timeout_s,
            unary_operators=config.unary_operators,
            binary_operators=config.binary_operators,
            layer_index=layer_index,
            out_index=o,
            in_index=i,
            n_points=config.n_points,
            domain=config.domain,
        )
        results.append(res)
    return results



