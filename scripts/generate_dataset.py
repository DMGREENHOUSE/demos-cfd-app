
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from demos_cfd_app.synthetic_model import compute_velocity_profiles

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, default="turbulence_intensity,inlet_temperature,inlet_velocity,outlet_pressure,wall_roughness")
    p.add_argument("--mins", type=str, default="0,250,0,0,0")
    p.add_argument("--maxs", type=str, default="20,400,50,100000,1.0")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("dataset.csv"))
    args = p.parse_args()

    names = [s.strip() for s in args.inputs.split(",") if s.strip()]
    mins  = [float(s) for s in args.mins.split(",")]
    maxs  = [float(s) for s in args.maxs.split(",")]
    if not (len(names) == len(mins) == len(maxs)):
        raise SystemExit("inputs, mins, and maxs must have same length")

    rng = np.random.default_rng(args.seed)
    data = {n: rng.uniform(a, b, size=args.n) for n, a, b in zip(names, mins, maxs)}
    X = pd.DataFrame(data)
    Y = compute_velocity_profiles(X, M=args.M, rng=rng)
    df = pd.concat([X, Y], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
