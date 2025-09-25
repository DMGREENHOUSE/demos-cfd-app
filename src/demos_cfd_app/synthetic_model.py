
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

def _col(X: pd.DataFrame, name: str, default: float = None) -> np.ndarray:
    if name in X.columns:
        return X[name].to_numpy().astype(float).reshape(-1)
    if default is None:
        return np.zeros((len(X),), dtype=float)
    return np.full((len(X),), float(default), dtype=float)

def compute_velocity_profiles(
    X: pd.DataFrame,
    *,
    M: int,
    spatial_amp1: float = 0.15,
    spatial_amp2: float = 0.10,
    interaction_strength: float = 0.10,
    noise_low: float = 0.0,
    noise_mid: float = 0.0,
    noise_high: float = 0.0,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Synthetic CFD-like velocity profiles with simple, reasonable dependencies on ANSYS-style inputs.
    """
    N = len(X)
    if rng is None:
        rng = np.random.default_rng(42)

    U_in = _col(X, "inlet_velocity", default=10.0)          # m/s
    P_out = _col(X, "outlet_pressure", default=0.0)         # Pa
    k_s = _col(X, "wall_roughness", default=0.0)            # mm
    TI = _col(X, "turbulence_intensity", default=5.0)       # %
    T_in = _col(X, "inlet_temperature", default=300.0)      # K

    # Nonlinear saturation with inlet velocity (mimics regime change)
    U_mid = np.maximum(1.0, np.median(U_in))
    U_core = np.tanh(U_in / (0.6 * U_mid + 1e-9))  # 0..~1

    # Effects
    beta_p = 0.35
    backpressure = np.clip(1.0 - beta_p * (P_out / 1e5), 0.4, 1.1)

    gamma_r = 0.6
    # k_s is in mm; treat 1.0 mm as strong roughness
    rough_scale = np.clip(1.0 - gamma_r * (k_s / 1.0), 0.5, 1.0)

    eta_T = 0.10
    temp_scale = 1.0 - eta_T * ((T_in - 300.0) / 100.0)

    mag = np.clip(U_core * backpressure * rough_scale * temp_scale, 0.0, None)

    xloc = np.linspace(0.05, 0.95, M).reshape(1, -1)
    damp_TI = np.clip(1.0 - 0.8 * (TI / 20.0), 0.2, 1.0).reshape(-1, 1)
    spatial = 1.0 + damp_TI * (spatial_amp1 * np.sin(2 * np.pi * xloc) + spatial_amp2 * np.cos(4 * np.pi * xloc))

    base = mag.reshape(-1, 1) * spatial

    # Noise support (defaults to zero)
    if (noise_low > 0) or (noise_mid > 0) or (noise_high > 0):
        std = np.full((N,), noise_low, dtype=float)
        std += (noise_mid - noise_low) * np.exp(-0.5 * ((U_in - U_mid) / (0.3 * (np.std(U_in) + 1e-9))) ** 2)
        std += noise_high * ((U_in - U_in.min() + 1e-9) / (U_in.max() - U_in.min() + 1e-9)) ** 0.25
        noise = rng.normal(size=base.shape) * std.reshape(-1, 1)
        Y = base + noise
    else:
        Y = base

    cols = [f"u_{i}" for i in range(M)]
    return pd.DataFrame(Y, columns=cols)
