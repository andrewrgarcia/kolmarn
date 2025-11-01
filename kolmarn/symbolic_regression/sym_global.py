from __future__ import annotations
import torch
import numpy as np

from .helpers import _discover_symbolic_batch
from .components import GlobalSymbolicConfig, _HAS_PYSR, _HAS_SYMPY
if _HAS_PYSR:
    from pysr import PySRRegressor
if _HAS_SYMPY:
    import sympy as sp


@torch.no_grad()
def discover_symbolic_global(
    model: torch.nn.Module,
    *,
    config: GlobalSymbolicConfig = GlobalSymbolicConfig(),
):
    """
    Stage-2 (Option A): run PySR on the full model output f_model(X).
    Produces analytic surrogates for each output dimension.
    """
    if not _HAS_PYSR:
        raise ImportError("PySR not installed. Install with `pip install pysr`.")

    model.eval()
    device = config.device or next(model.parameters()).device

    try:
        in_features = getattr(model[0], "in_features", 1)
    except Exception:
        in_features = 1

    # Normalize domain shape
    domain = config.domain
    if isinstance(domain[0], (int, float)):
        domain = [domain for _ in range(in_features)]
    else:
        domain = list(domain)

    X_np = np.zeros((config.n_samples, in_features))
    for j, (lo, hi) in enumerate(domain):
        X_np[:, j] = lo + (hi - lo) * np.random.rand(config.n_samples)

    # Clip to avoid log(0) or divide-by-zero issues
    X_np = np.clip(X_np, 1e-6, None)

    X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    y_np = model(X).detach().cpu().numpy()
    if y_np.ndim == 1:
        y_np = y_np[:, None]

    results = _discover_symbolic_batch(
        X_np,
        y_np,
        meta_fn=lambda j: {
            "layer_index": -1,
            "out_index": j,
            "in_index": -1,
            "n_points": config.n_samples,
            "domain": (
                float(np.min(X_np)),
                float(np.max(X_np)),
            ),
        },
        maxsize=config.maxsize,
        niterations=config.niterations,
        timeout_s=config.timeout_s,
        unary_operators=config.unary_operators,
        binary_operators=config.binary_operators,
    )

    return results
