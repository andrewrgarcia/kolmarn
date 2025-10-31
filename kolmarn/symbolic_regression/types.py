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
    layer_index: int
    out_index: int
    in_index: int
    expr_sympy: Optional["sp.Expr"]          # sympy expression or None
    expr_str: Optional[str]                  # stringified expression
    r2: float
    rmse: float
    length: Optional[int]                    # expression length/complexity if available
    n_points: int
    x_domain: Tuple[float, float]            # (xmin, xmax)
    notes: Optional[str] = None

    def to_latex(self) -> str:
        if not _HAS_SYMPY or self.expr_sympy is None:
            return r"\text{(sympy unavailable or expression is None)}"
        return sp.latex(self.expr_sympy)

    def to_callable_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a numpy-callable f(x)."""
        if not _HAS_SYMPY or self.expr_sympy is None:
            raise RuntimeError("Symbolic expression not available. Install sympy or run discovery with PySR.")

        x = sp.Symbol("x")
        f_np = sp.lambdify(x, self.expr_sympy, modules="numpy")
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