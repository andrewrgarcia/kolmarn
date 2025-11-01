from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .variables import _HAS_SYMPY
if _HAS_SYMPY:
    import sympy as sp


@dataclass
class SymbolicResult:
    """
    Container for a single symbolic regression result.

    Attributes
    ----------
    layer_index : int
        Index of the KAN layer from which the relation was extracted.
    out_index : int
        Output neuron index within the layer.
    in_index : int
        Input neuron index within the layer.
    expr_sympy : Optional[sympy.Expr]
        Discovered symbolic expression as a SymPy object, or None if unavailable.
    expr_str : Optional[str]
        Stringified version of the discovered expression.
    r2 : float
        Coefficient of determination between symbolic prediction and true data.
    rmse : float
        Root-mean-square error between symbolic prediction and true data.
    length : Optional[int]
        Symbolic expression complexity (operator length), if available.
    n_points : int
        Number of data points used during symbolic discovery.
    x_domain : Tuple[float, float]
        Sampling domain (xmin, xmax) for the unit.
    notes : Optional[str]
        Optional free-form notes or metadata returned by the backend.

    noise_std : Optional[float]
        Estimated residual noise standard deviation (from `estimate_noise`).
    pruned_expr : Optional[sympy.Expr]
        Expression after coefficient simplification (`prune`).
    expr_stability : Optional[float]
        0-1 stability score across ensemble runs (`summarize_ensemble`).
    expr_variants : Optional[list[str]]
        List of raw expression strings discovered across ensemble runs.

    runtime_s : Optional[float]
        Wall-clock seconds for symbolic discovery.
    start_time : Optional[float]
        Absolute start timestamp, useful for logging/debugging.
    """
    layer_index: int
    out_index: int
    in_index: int
    expr_sympy: Optional["sp.Expr"] 
    expr_str: Optional[str]
    r2: float
    rmse: float
    length: Optional[int]
    n_points: int
    x_domain: Tuple[float, float]
    notes: Optional[str] = None
    noise_std: Optional[float] = None
    pruned_expr: Optional["sp.Expr"] = None
    expr_stability: Optional[float] = None 
    expr_variants: Optional[list[str]] = None
    runtime_s: Optional[float] = None
    start_time: Optional[float] = None

    def to_latex(self) -> str:
        if not _HAS_SYMPY or self.expr_sympy is None:
            return r"\text{(sympy unavailable or expression is None)}"
        return sp.latex(self.expr_sympy)

    def to_callable_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a numpy-callable f(x)."""
        if not _HAS_SYMPY or self.expr_sympy is None:
            raise RuntimeError("Sympy expression not available.")

        # Normalize variable names
        expr = self.expr_sympy
        free_syms = list(expr.free_symbols)
        if len(free_syms) > 0:
            x_sym = free_syms[0]
        else:
            x_sym = sp.Symbol("x")

        # Force numeric-safe mapping
        f_np = sp.lambdify(x_sym, expr, modules=["numpy", {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log}])

        return lambda arr: np.asarray(f_np(arr), dtype=float)

    def to_callable_torch(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a torch-callable f(x) that works on 1D or 2D tensors (..., 1)."""
        f_np = self.to_callable_numpy()
        def f_torch(x: torch.Tensor) -> torch.Tensor:
            # Ensure CPU numpy conversion and back
            x_np = x.detach().cpu().numpy()
            y_np = f_np(x_np)
            y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
            # Preserve device/batch shape
            y = y.to(x.device)
            return y
        return f_torch
    
    def estimate_noise(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute residual std between data and symbolic prediction."""
        try:
            f_np = self.to_callable_numpy()
            y_hat = f_np(X)
            self.noise_std = float(np.std(y - y_hat))
        except Exception:
            self.noise_std = float("nan")
        return self.noise_std

    def prune(self, tol: float = 1e-3):
        """Simplify small coefficients within tolerance."""
        if not _HAS_SYMPY or self.expr_sympy is None:
            return self
        try:
            self.pruned_expr = sp.nsimplify(self.expr_sympy, tolerance=tol)
        except Exception:
            self.pruned_expr = self.expr_sympy
        return self
    
    def summarize_ensemble(self, results: list["SymbolicResult"]) -> "SymbolicResult":
        """
        Compute expression stability across ensemble runs.
        Stability = 1 - (variance in operator composition or coefficient pattern).
        """
        if not results:
            return self
        exprs = [r.expr_str for r in results if r.expr_str]
        self.expr_variants = exprs

        # Very simple operator-level stability
        ops = ["sin", "cos", "exp", "log"]
        counts = np.array([[e.count(op) for op in ops] for e in exprs])
        if counts.size == 0:
            self.expr_stability = 0.0
        else:
            # normalized variability
            var = np.mean(np.std(counts, axis=0) / (np.mean(counts, axis=0) + 1e-6))
            self.expr_stability = float(max(0.0, 1.0 - var))
        return self
