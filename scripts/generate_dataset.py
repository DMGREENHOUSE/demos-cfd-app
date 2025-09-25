
import argparse
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for local runs without install
HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from demos_cfd_app.dataset import generate_dataset

def float_pair(s: str):
    a, b = s.split(",")
    return float(a), float(b)

def main():
    p = argparse.ArgumentParser(description="Generate synthetic CFD-like dataset and save as CSV.")
    p.add_argument("--inputs", type=str, default="Re", help="Comma-separated input names, e.g., 'Re,PR,Tin'")
    p.add_argument("--mins", type=str, default="1e3", help="Comma-separated mins, e.g., '1e3,0,250'")
    p.add_argument("--maxs", type=str, default="5e5", help="Comma-separated maxs, e.g., '5e5,1,400'")
    p.add_argument("--n", type=int, default=200, help="Number of points (for random/grid sampling)")
    p.add_argument("--sampling", type=str, default="random", choices=["random", "grid"], help="Sampling mode")
    p.add_argument("--inputs-csv", type=Path, default=None, help="Optional path to CSV of inputs to evaluate")
    p.add_argument("--gap-var", type=str, default=None, help="Variable name for gaps (optional)")
    p.add_argument("--gap", action="append", default=[], help="Gap interval 'a,b'. Can be repeated.")
    p.add_argument("--M", type=int, default=10, help="Number of velocity locations")
    p.add_argument("--primary", type=str, default=None, help="Primary driver variable (defaults to first input)")
    p.add_argument("--plateaus-on", action="store_true", help="Enable plateau behaviour")
    p.add_argument("--plateau-low", type=str, default=None, help="Low plateau 'a,b' (defaults to 20% from min)")
    p.add_argument("--plateau-high", type=str, default=None, help="High plateau 'a,b' (defaults to top 30%)")
    p.add_argument("--transition-center", type=float, default=None)
    p.add_argument("--transition-width", type=float, default=None)
    p.add_argument("--spatial-amp1", type=float, default=0.15)
    p.add_argument("--spatial-amp2", type=float, default=0.10)
    p.add_argument("--interaction-strength", type=float, default=0.10)
    p.add_argument("--noise-low", type=float, default=0.01)
    p.add_argument("--noise-mid", type=float, default=0.08)
    p.add_argument("--noise-high", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("dataset.csv"))
    args = p.parse_args()

    names = [s.strip() for s in args.inputs.split(",") if s.strip()]
    mins  = [float(s) for s in args.mins.split(",")]
    maxs  = [float(s) for s in args.maxs.split(",")]
    if not (len(names) == len(mins) == len(maxs)):
        raise SystemExit("inputs, mins, and maxs must have the same length")

    ranges = {n: (a, b) for n, a, b in zip(names, mins, maxs)}

    inputs_df = None
    if args.inputs_csv is not None:
        inputs_df = pd.read_csv(args.inputs_csv)

    gaps = [float_pair(g) for g in args.gap] if args.gap else []

    low = float_pair(args.plateau_low) if args.plateau_low else None
    high = float_pair(args.plateau_high) if args.plateau_high else None

    df = generate_dataset(
        n=args.n,
        ranges=ranges,
        sampling=args.sampling,
        inputs_df=inputs_df,
        gap_var=args.gap_var,
        gaps=gaps,
        M=args.M,
        primary_var=args.primary or names[0],
        plateaus_on=args.plateaus_on,
        plateau_low=low,
        plateau_high=high,
        transition_center=args.transition_center,
        transition_width=args.transition_width,
        spatial_amp1=args.spatial_amp1,
        spatial_amp2=args.spatial_amp2,
        interaction_strength=args.interaction_strength,
        noise_low=args.noise_low,
        noise_mid=args.noise_mid,
        noise_high=args.noise_high,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
