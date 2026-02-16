from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(values: list[np.ndarray]) -> dict[str, float]:
    if not values:
        return {}
    x = np.concatenate(values)
    if x.size == 0:
        return {}
    return {
        "min": float(np.min(x)),
        "p1": float(np.quantile(x, 0.01)),
        "p5": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def main() -> None:
    root = Path("ML/datasets/Toro Ossaba")
    files = [p for p in root.rglob("*.txt") if p.name != "subject_info.txt"]
    raw1: list[np.ndarray] = []
    raw2: list[np.ndarray] = []
    fil1: list[np.ndarray] = []
    fil2: list[np.ndarray] = []

    for p in files:
        df = pd.read_csv(p, header=None, sep=r"\s+")
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.shape[1] < 5:
            continue
        arr = df.to_numpy(dtype=float, copy=False)
        raw1.append(arr[:, 0])
        raw2.append(arr[:, 1])
        fil1.append(arr[:, 2])
        fil2.append(arr[:, 3])

    out = {
        "raw_ch1": summarize(raw1),
        "raw_ch2": summarize(raw2),
        "filtered_ch1": summarize(fil1),
        "filtered_ch2": summarize(fil2),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
