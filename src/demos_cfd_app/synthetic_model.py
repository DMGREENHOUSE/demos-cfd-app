
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

def compute_velocity_profiles(
    X: pd.DataFrame,
    *,
    M: int,
    primary_var: str,
    plateaus_on: bool,
    plateau_low: Tuple[float, float],
    plateau_high: Tuple[float, float],
    transition_center: float,
    transition_width: float,
    spatial_amp1: float,
    spatial_amp2: float,
    interaction_strength: float,
    noise_low: float,
    noise_mid: float,
    noise_high: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Synthetic CFD-like velocity profiles u_0..u_{M-1} as a function of inputs X.
    - 'primary_var' drives a tanh transition between low/high plateaus.
    - Other variables apply a mild linear scale interaction.
    - Heteroscedastic noise is larger near the transition.
    """
    assert primary_var in X.columns, "primary_var must be one of the input columns"
    v = X[primary_var].to_numpy().astype(float).reshape(-1)

    # Smooth transition (low -> high) across primary_var
    t = 0.5 * (1.0 + np.tanh((v - transition_center) / max(1e-9, transition_width)))
    low_val, high_val = 0.4, 1.0
    core = (1 - t) * low_val + t * high_val  # (N,)

    # Spatial structure for M outputs
    xloc = np.linspace(0.05, 0.95, M).reshape(1, -1)
    spatial = 1.0 + spatial_amp1 * np.sin(2 * np.pi * xloc) + spatial_amp2 * np.cos(4 * np.pi * xloc)
    base = core.reshape(-1, 1) * spatial  # (N, M)

    # Plateaus: clamp within given bands
    if plateaus_on:
        mask_low = (v >= min(*plateau_low)) & (v <= max(*plateau_low))
        mask_high = (v >= min(*plateau_high)) & (v <= max(*plateau_high))
        if mask_low.any():
            base[mask_low] = base[mask_low][0]
        if mask_high.any():
            base[mask_high] = base[mask_high][0]

    # Linear interaction from other variables (normalized [-0.5, 0.5])
    extra_cols = [c for c in X.columns if c != primary_var]
    if extra_cols and interaction_strength != 0.0:
        effects = np.zeros((len(X), 1))
        for c in extra_cols:
            arr = X[c].to_numpy().astype(float).reshape(-1)
            a, b = float(np.nanmin(arr)), float(np.nanmax(arr))
            norm = (arr - a) / (b - a) - 0.5 if b > a else np.zeros_like(arr)
            effects += interaction_strength * norm.reshape(-1, 1)
        base = base * (1.0 + effects)

    # Heteroscedastic noise: bump near transition + light tail
    std = np.full_like(v, noise_low, dtype=float)
    std += (noise_mid - noise_low) * np.exp(-0.5 * ((v - transition_center) / max(1e-9, transition_width)) ** 2)
    denom = (v.max() - v.min() + 1e-9)
    std += noise_high * ((v - v.min() + 1e-9) / denom) ** 0.25

    noise = rng.normal(size=base.shape) * std.reshape(-1, 1)
    Y = base + noise

    cols = [f"u_{i}" for i in range(M)]
    return pd.DataFrame(Y, columns=cols)
