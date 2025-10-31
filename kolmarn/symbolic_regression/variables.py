try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    _HAS_PYSR = False

try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False
