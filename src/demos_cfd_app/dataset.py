
from __future__ import annotations
import numpy as np
import pandas as pd
from .synthetic_model import compute_velocity_profiles

def generate_dataset(X: pd.DataFrame, *, M: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Y = compute_velocity_profiles(X, M=M, rng=rng)
    return pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
