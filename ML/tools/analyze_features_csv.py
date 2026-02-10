from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_features_csv(path: Path, *, chunksize: int = 200_000) -> None:
    if not path.is_file():
        raise SystemExit(f"features.csv not found: {path}")

    cols = {"subject", "record", "label", "vel_norm", "vel_dps"}

    label_counts: dict[int, int] = {}
    vel_n = 0
    vel_clip = 0
    vel_sum = 0.0
    vel_abs_sum = 0.0
    vel_min = float("inf")
    vel_max = float("-inf")
    # histogram bins for vel_norm in [-1, 1] with width 0.05
    hist = np.zeros(41, dtype=np.int64)

    # Per-record velocity clip % can reveal if some recordings are pathological
    per_record: dict[str, dict[str, float]] = {}  # {record: {"n":..., "clip":...}}

    def get_rec(rec_val: object) -> str:
        s = str(rec_val)
        return s if s else "<empty>"

    for chunk in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=int(chunksize)):
        if "label" in chunk.columns:
            s = pd.to_numeric(chunk["label"], errors="coerce").dropna()
            if not s.empty:
                for v, c in zip(*np.unique(s.astype(int).to_numpy(), return_counts=True)):
                    label_counts[int(v)] = int(label_counts.get(int(v), 0) + int(c))

        if "vel_norm" in chunk.columns:
            v = pd.to_numeric(chunk["vel_norm"], errors="coerce").dropna()
            if not v.empty:
                vv = v.to_numpy(dtype=float, copy=False)
                vel_n += int(vv.size)
                vel_clip += int(np.sum(np.isclose(np.abs(vv), 1.0)))
                vel_sum += float(np.sum(vv))
                vel_abs_sum += float(np.sum(np.abs(vv)))
                vel_min = min(vel_min, float(np.min(vv)))
                vel_max = max(vel_max, float(np.max(vv)))

                idx = np.floor((np.clip(vv, -1.0, 1.0) + 1.0) / 0.05).astype(int)
                idx = np.clip(idx, 0, 40)
                hist += np.bincount(idx, minlength=41).astype(np.int64)

                if "record" in chunk.columns:
                    recs = chunk["record"].astype(str).to_numpy()
                    clipped = np.isclose(np.abs(vv), 1.0)
                    for r in np.unique(recs):
                        m = recs == r
                        n = int(np.sum(m))
                        if n == 0:
                            continue
                        c = int(np.sum(clipped[m]))
                        d = per_record.setdefault(get_rec(r), {"n": 0.0, "clip": 0.0})
                        d["n"] += float(n)
                        d["clip"] += float(c)

    print("=== features.csv quick stats ===")
    if label_counts:
        print("label_counts:", {k: int(label_counts[k]) for k in sorted(label_counts)})
    else:
        print("label_counts: (none found)")

    if vel_n:
        print(
            "vel_norm:",
            {
                "n": int(vel_n),
                "min": float(vel_min),
                "max": float(vel_max),
                "mean": float(vel_sum / vel_n),
                "mean_abs": float(vel_abs_sum / vel_n),
                "clip_%": float(vel_clip / vel_n),
            },
        )

        top = np.argsort(hist)[-12:][::-1]
        print("top vel_norm bins (center -> count):")
        for i in top:
            center = -1.0 + 0.05 * float(i)
            print(f"  {center:+.2f} -> {int(hist[i])}")

        # Show the most clipped recordings (if any)
        if per_record:
            rows = []
            for r, d in per_record.items():
                n = float(d["n"])
                c = float(d["clip"])
                if n > 0:
                    rows.append((c / n, int(c), int(n), r))
            rows.sort(reverse=True)
            print("most clipped records (clip_% , clipped , n , record):")
            for frac, c, n, r in rows[:10]:
                print(f"  {frac:6.2%} {c:7d} {n:7d} {r}")
    else:
        print("vel_norm: (none found)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        type=str,
        default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled/features.csv",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    analyze_features_csv(Path(args.features_csv), chunksize=int(args.chunksize))


if __name__ == "__main__":
    main()

