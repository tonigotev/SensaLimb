from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordStats:
    subject: str
    record: str
    n: int
    angle_min: float
    angle_max: float
    angle_mean: float
    angle_std: float
    unique_count: int
    unique_ratio: float
    median_step: float
    mode_step: float
    plateau_mean_len: float


def find_labeled_csvs(root: Path) -> list[Path]:
    files = sorted([p for p in root.rglob("*.csv") if p.is_file() and p.name.endswith("_labeled.csv")])
    if not files:
        raise SystemExit(f"No *_labeled.csv files found under: {root}")
    return files


def infer_subject_record(p: Path) -> tuple[str, str]:
    return p.parent.name, p.stem


def read_angle_column(csv_path: Path, sample_step: int) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None, usecols=[2])
    a = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if sample_step > 1:
        a = a[::sample_step]
    return a


def median_and_mode_steps(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    uniq = np.unique(values)
    if uniq.size < 2:
        return 0.0, 0.0
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0, 0.0
    median_step = float(np.median(diffs))
    # approximate mode by binning diff values
    bins = np.round(diffs / max(median_step, 1e-9)).astype(np.int64)
    mode_bin = int(np.bincount(bins).argmax())
    mode_step = float(mode_bin * median_step)
    return median_step, mode_step


def plateau_stats(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    # run-lengths of constant values
    changes = np.where(np.diff(values) != 0)[0] + 1
    runs = np.diff(np.concatenate(([0], changes, [values.size])))
    return float(np.mean(runs)) if runs.size else float(values.size)


def compute_record_stats(csv_path: Path, sample_step: int) -> RecordStats:
    subject, record = infer_subject_record(csv_path)
    a = read_angle_column(csv_path, sample_step=sample_step)
    if a.size == 0:
        return RecordStats(subject, record, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0.0, float("nan"), float("nan"), float("nan"))
    uniq = np.unique(a)
    median_step, mode_step = median_and_mode_steps(a)
    plateau_mean = plateau_stats(a)
    return RecordStats(
        subject=subject,
        record=record,
        n=int(a.size),
        angle_min=float(np.min(a)),
        angle_max=float(np.max(a)),
        angle_mean=float(np.mean(a)),
        angle_std=float(np.std(a)),
        unique_count=int(uniq.size),
        unique_ratio=float(uniq.size / max(a.size, 1)),
        median_step=median_step,
        mode_step=mode_step,
        plateau_mean_len=plateau_mean,
    )


def summarize(stats: list[RecordStats]) -> dict:
    if not stats:
        return {}
    vals = np.array([s.angle_std for s in stats if np.isfinite(s.angle_std)], dtype=np.float64)
    uniq_ratio = np.array([s.unique_ratio for s in stats if np.isfinite(s.unique_ratio)], dtype=np.float64)
    med_step = np.array([s.median_step for s in stats if np.isfinite(s.median_step)], dtype=np.float64)
    plateau = np.array([s.plateau_mean_len for s in stats if np.isfinite(s.plateau_mean_len)], dtype=np.float64)
    return {
        "records": len(stats),
        "angle_std_mean": float(np.mean(vals)) if vals.size else float("nan"),
        "angle_std_median": float(np.median(vals)) if vals.size else float("nan"),
        "unique_ratio_mean": float(np.mean(uniq_ratio)) if uniq_ratio.size else float("nan"),
        "unique_ratio_median": float(np.median(uniq_ratio)) if uniq_ratio.size else float("nan"),
        "median_step_mean": float(np.mean(med_step)) if med_step.size else float("nan"),
        "median_step_median": float(np.median(med_step)) if med_step.size else float("nan"),
        "plateau_mean_len_mean": float(np.mean(plateau)) if plateau.size else float("nan"),
        "plateau_mean_len_median": float(np.median(plateau)) if plateau.size else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sanity check angle signal quality in labeled CSVs.")
    p.add_argument("--input-root", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled")
    p.add_argument("--only-subject", type=str, default=None)
    p.add_argument("--sample-step", type=int, default=1, help="Subsample angle series by this step to speed up stats.")
    p.add_argument("--max-records", type=int, default=0, help="Limit number of records (0 = all).")
    p.add_argument("--out-json", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/angle_sanity.json")
    args = p.parse_args()

    root = Path(args.input_root)
    files = find_labeled_csvs(root)
    if args.only_subject:
        key = str(args.only_subject).casefold()
        files = [p for p in files if p.parent.name.casefold() == key or p.parent.name.replace("Subject_", "subject_").casefold() == key]
    if args.max_records and int(args.max_records) > 0:
        files = files[: int(args.max_records)]

    stats = [compute_record_stats(p, sample_step=int(args.sample_step)) for p in files]
    summary = summarize(stats)

    payload = {
        "input_root": str(root),
        "only_subject": str(args.only_subject) if args.only_subject else None,
        "sample_step": int(args.sample_step),
        "summary": summary,
        "records": [s.__dict__ for s in stats],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

