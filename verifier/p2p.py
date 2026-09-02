"""P2P (pass-to-pass) regression penalty candidates.

multiplier(n_pass, n_total, ...) -> factor in (0,1] applied to the F2P credit. k = observed
regressions among n checked tests (n is capped at 30 upstream, so observed failures are certain
evidence while unobserved ones are unknown: penalties are conservative for k<=2 and strong for
catastrophic regressions).
"""
from __future__ import annotations

import math


def wilson_lower(k: int, n: int, z: float = 1.0) -> float:
    """Lower bound of the Wilson score interval for the regression rate k/n."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def multiplier(n_pass: int, n_total: int, mode: str = "pow", p2p_max: float = 0.75, gamma: float = 1.5,
               tolerance: int = 0, cap: float = 0.5, z: float = 1.0, catastrophic_frac: float = 0.5,
               catastrophic_mult: float = 0.25) -> float:
    if n_total <= 0 or mode == "none":
        return 1.0
    k = max(0, n_total - n_pass)
    reg = k / n_total
    if mode == "linear":
        return max(0.0, 1.0 - p2p_max * reg)
    if mode == "pow":                      # baseline: 1 - 0.75 * reg^1.5
        return max(0.0, 1.0 - p2p_max * reg ** gamma)
    if mode == "capped":                   # linear but the penalty never exceeds `cap`
        return max(0.0, 1.0 - min(cap, p2p_max * reg))
    if mode == "tolerance":                # first `tolerance` regressions are free (flaky allowance)
        reg2 = max(0, k - tolerance) / n_total
        return max(0.0, 1.0 - p2p_max * reg2 ** gamma)
    if mode == "confidence":               # penalise only the statistically certain part
        lb = wilson_lower(k, n_total, z)
        return max(0.0, 1.0 - p2p_max * lb ** gamma)
    if mode == "catastrophic":             # tolerance for small k, hard multiplier past a threshold
        if reg >= catastrophic_frac:
            return catastrophic_mult
        reg2 = max(0, k - tolerance) / n_total
        return max(0.0, 1.0 - p2p_max * reg2 ** gamma)
    raise ValueError(f"unknown p2p mode {mode}")
