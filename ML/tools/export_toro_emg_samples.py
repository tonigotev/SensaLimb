from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Export fixed-count EMG samples for Nucleo ADC simulation.")
    p.add_argument("--input-file", type=str, required=True, help="Toro Ossaba .txt file path.")
    p.add_argument("--output-file", type=str, required=True, help="Output file path (csv).")
    p.add_argument("--channels", choices=["raw", "filtered"], default="filtered")
    p.add_argument("--n-samples", type=int, default=1000)
    args = p.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    df = pd.read_csv(in_path, header=None, sep=r"\s+")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 5:
        raise SystemExit("Expected at least 5 columns in input file.")

    if args.channels == "raw":
        data = df.iloc[: int(args.n_samples), 0:2].to_numpy(dtype=np.float32)
    else:
        data = df.iloc[: int(args.n_samples), 2:4].to_numpy(dtype=np.float32)

    if data.shape[0] < int(args.n_samples):
        raise SystemExit(f"Not enough rows ({data.shape[0]}) for requested n-samples={args.n_samples}.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, data, delimiter=",", fmt="%.6f")
    print(f"Saved {data.shape[0]} samples to {out_path}")


if __name__ == "__main__":
    main()
