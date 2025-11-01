from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

@dataclass
class SymbolicConfig:
    n_points: int = 512
    domain: Tuple[float, float] = (0.0, 1.0)
    method: Literal["pysr"] = "pysr"
    maxsize: int = 12
    niterations: int = 2000
    timeout_s: Optional[int] = None
    unary_operators: Optional[List[str]] = None
    binary_operators: Optional[List[str]] = None


@dataclass
class GlobalSymbolicConfig(SymbolicConfig):
    """
    Extension of SymbolicConfig for global SR (model-level).
    Adds sampling and device control.
    """
    n_samples: int = 2048
    device: Optional[str] = None
