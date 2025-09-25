
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from .synthetic_model import compute_velocity_profiles

def apply_gaps(X: pd.DataFrame, gap_var: Optional[str], gaps: Optional[List[Tuple[float, float]]]) -> pd.DataFrame:
    """Exclude rows where gap_var lies within any [a, b] interval in gaps."""
    if gap_var is None or not gaps:
        return X
    mask = np.ones(len(X), dtype=bool)
    v = X[gap_var].to_numpy().astype(float)
    for a, b in gaps:
        lo, hi = min(a, b), max(a, b)
        mask &= ~((v >= lo) & (v <= hi))
    return X[mask].reset_index(drop=True)

def random_sample(n: int, ranges: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> pd.DataFrame:
    data = {k: rng.uniform(a, b, size=n) for k, (a, b) in ranges.items()}
    return pd.DataFrame(data)

def grid_sample(n_target: int, ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Structured grid with ~n_target points spread across dims evenly."""
    keys = list(ranges.keys())
    d = len(keys)
    steps = max(2, int(round(n_target ** (1 / max(1, d)))))
    axes = []
    for k in keys:
        a, b = ranges[k]
        axes.append(np.linspace(a, b, steps))
    grids = np.meshgrid(*axes, indexing="xy")
    flat = [g.reshape(-1) for g in grids]
    return pd.DataFrame({k: v for k, v in zip(keys, flat)})

def generate_dataset(
    *,
    n: int,
    ranges: Dict[str, Tuple[float, float]],
    sampling: str = "random",
    inputs_df: Optional[pd.DataFrame] = None,
    gap_var: Optional[str] = None,
    gaps: Optional[List[Tuple[float, float]]] = None,
    M: int = 10,
    primary_var: str = None,
    plateaus_on: bool = True,
    plateau_low: Tuple[float, float] = None,
    plateau_high: Tuple[float, float] = None,
    transition_center: float = None,
    transition_width: float = None,
    spatial_amp1: float = 0.15,
    spatial_amp2: float = 0.10,
    interaction_strength: float = 0.1,
    noise_low: float = 0.01,
    noise_mid: float = 0.08,
    noise_high: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a combined dataset of inputs and velocity outputs.
    If inputs_df is provided, it is used directly (after applying gaps). Otherwise, sample from 'ranges'.
    """
    rng = np.random.default_rng(seed)

    # Prepare inputs
    if inputs_df is not None:
        X = inputs_df.copy()
    else:
        if sampling.lower().startswith("grid"):
            X = grid_sample(n, ranges)
        else:
            X = random_sample(n, ranges, rng)

    # Set defaults from ranges if not given
    if primary_var is None:
        primary_var = list(ranges.keys())[0]
    pr_a, pr_b = ranges[primary_var]
    if transition_center is None:
        transition_center = 0.5 * (pr_a + pr_b)
    if transition_width is None:
        transition_width = max(1e-6, 0.1 * (pr_b - pr_a))
    if plateau_low is None:
        plateau_low = (pr_a, pr_a + 0.2*(pr_b - pr_a))
    if plateau_high is None:
        plateau_high = (pr_a + 0.7*(pr_b - pr_a), pr_b)

    # Apply gaps
    X = apply_gaps(X, gap_var, gaps)

    # Compute outputs
    Y = compute_velocity_profiles(
        X,
        M=M,
        primary_var=primary_var,
        plateaus_on=plateaus_on,
        plateau_low=plateau_low,
        plateau_high=plateau_high,
        transition_center=transition_center,
        transition_width=transition_width,
        spatial_amp1=spatial_amp1,
        spatial_amp2=spatial_amp2,
        interaction_strength=interaction_strength,
        noise_low=noise_low,
        noise_mid=noise_mid,
        noise_high=noise_high,
        rng=rng,
    )

    return pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
