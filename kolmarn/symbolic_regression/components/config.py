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
    Adds sampling, device control, and runtime presets via `mode`.
    """
    n_samples: int = 2048
    device: Optional[str] = None
    ensemble_runs: int = 5
    ensemble_perturb: float = 0.02
    tolerance: float = 1e-3
    mode: Literal["fast", "full"] = "full"

    def __post_init__(self):
        """Apply predefined speed/quality trade-offs."""
        if self.mode == "fast":
            # Fast development mode
            self.n_samples = 1024
            self.niterations = 800
            self.maxsize = 10
            self.timeout_s = 40
            self.ensemble_runs = 3
            self.ensemble_perturb = 0.01
        elif self.mode == "full":
            # Full research-quality mode
            self.n_samples = 4096
            self.niterations = 3000
            self.maxsize = 12
            self.timeout_s = 120
            self.ensemble_runs = 5
            self.ensemble_perturb = 0.02
