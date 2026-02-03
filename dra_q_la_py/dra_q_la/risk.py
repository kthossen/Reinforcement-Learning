
import numpy as np

def risk_mean(x):
    return float(np.mean(x))

def risk_min(x):
    return float(np.min(x))

def risk_max(x):
    return float(np.max(x))

def var(x, alpha=0.25):
    """Value-at-Risk at level alpha (lower tail)."""
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, alpha, method="linear"))

def cvar(x, alpha=0.25):
    """Conditional VaR (Expected Shortfall) for lower tail."""
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, alpha, method="linear")
    return float(x[x <= q].mean() if np.any(x <= q) else q)
