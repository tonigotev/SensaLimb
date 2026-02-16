from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    subject: str
    record: str
    n_samples: int


def find_labeled_csvs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input root not found: {root}")
    files = sorted([p for p in root.rglob("*.csv") if p.is_file() and p.name.endswith("_labeled.csv")])
    if not files:
        raise SystemExit(f"No *_labeled.csv files found under: {root}")
    return files


def infer_subject_record(p: Path) -> tuple[str, str]:
    subj = p.parent.name
    rec = p.stem
    return subj, rec


def count_rows_fast(csv_path: Path) -> int:
    try:
        return int(pd.read_csv(csv_path, header=None, usecols=[0]).shape[0])
    except Exception:
        return int(pd.read_csv(csv_path, header=None).shape[0])


def build_record_index(files: list[Path]) -> list[RecordInfo]:
    out: list[RecordInfo] = []
    for p in files:
        subj, rec = infer_subject_record(p)
        n = count_rows_fast(p)
        out.append(RecordInfo(path=p, subject=subj, record=rec, n_samples=n))
    return out


def filter_only_subject(recs: list[RecordInfo], only_subject: str | None) -> list[RecordInfo]:
    if not only_subject:
        return recs
    key = str(only_subject).casefold()
    kept = [r for r in recs if r.subject.casefold() == key or r.subject.replace("Subject_", "subject_").casefold() == key]
    if not kept:
        raise SystemExit(f"No records match --only-subject {only_subject}. Example subjects: {sorted({r.subject for r in recs})[:10]}")
    return kept


def split_records(
    recs: list[RecordInfo],
    *,
    split_by: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[RecordInfo], list[RecordInfo], list[RecordInfo]]:
    rng = np.random.default_rng(seed)
    if split_by == "row":
        split_by = "record"

    if split_by == "record":
        items = recs[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(np.ceil(test_frac * n)) if test_frac > 0 else 0
        n_val = int(np.ceil(val_frac * n)) if val_frac > 0 else 0
        if test_frac > 0 and n >= 3:
            n_test = max(1, n_test)
        if val_frac > 0 and n >= 3:
            n_val = max(1, n_val)
        if (n_test + n_val) >= n:
            overflow = (n_test + n_val) - (n - 1)
            if overflow > 0 and n_val > 0:
                dec = min(overflow, n_val)
                n_val -= dec
                overflow -= dec
            if overflow > 0 and n_test > 0:
                dec = min(overflow, n_test)
                n_test -= dec
                overflow -= dec
        test = items[:n_test]
        val = items[n_test : n_test + n_val]
        train = items[n_test + n_val :]
        return train, val, test

    if split_by == "subject":
        subj = sorted({r.subject for r in recs})
        rng.shuffle(subj)
        n = len(subj)
        n_test = int(np.ceil(test_frac * n)) if test_frac > 0 else 0
        n_val = int(np.ceil(val_frac * n)) if val_frac > 0 else 0
        if test_frac > 0 and n >= 3:
            n_test = max(1, n_test)
        if val_frac > 0 and n >= 3:
            n_val = max(1, n_val)
        if (n_test + n_val) >= n:
            overflow = (n_test + n_val) - (n - 1)
            if overflow > 0 and n_val > 0:
                dec = min(overflow, n_val)
                n_val -= dec
                overflow -= dec
            if overflow > 0 and n_test > 0:
                dec = min(overflow, n_test)
                n_test -= dec
                overflow -= dec
        test_subj = set(subj[:n_test])
        val_subj = set(subj[n_test : n_test + n_val])
        train_subj = set(subj[n_test + n_val :])
        train = [r for r in recs if r.subject in train_subj]
        val = [r for r in recs if r.subject in val_subj]
        test = [r for r in recs if r.subject in test_subj]
        return train, val, test

    raise SystemExit(f"Unknown --split-by {split_by}")


@lru_cache(maxsize=8)
def load_labeled_array(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {csv_path}, got {arr.shape[1]}")
    if arr.shape[1] >= 4:
        arr = arr[:, :4]
    return arr


def window_envelope(x: np.ndarray, *, rms_win: int) -> np.ndarray:
    if rms_win <= 1:
        return np.abs(x)
    pad = rms_win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    sq = xp * xp
    c = np.cumsum(sq, axis=0)
    c[rms_win:] = c[rms_win:] - c[:-rms_win]
    ma = c[rms_win - 1 : -1] / float(rms_win)
    return np.sqrt(np.maximum(ma, 0.0))


def downsample(x: np.ndarray, *, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    n = (x.shape[0] // factor) * factor
    if n <= 0:
        return x[:0]
    xx = x[:n]
    xx = xx.reshape(n // factor, factor, x.shape[1])
    return np.mean(xx, axis=1)


def make_window_index(
    recs: list[RecordInfo],
    *,
    fs_hz: float,
    win_sec: float,
    step_sec: float,
    max_windows_per_record: int,
    seed: int,
) -> list[tuple[RecordInfo, int]]:
    rng = np.random.default_rng(seed)
    win = int(round(win_sec * fs_hz))
    step = int(round(step_sec * fs_hz))
    if win < 8 or step < 1:
        raise SystemExit("Invalid --win-sec/--step-sec for given --fs.")
    idx: list[tuple[RecordInfo, int]] = []
    for r in recs:
        starts = list(range(0, max(0, r.n_samples - win + 1), step))
        if not starts:
            continue
        if max_windows_per_record > 0 and len(starts) > max_windows_per_record:
            starts = rng.choice(np.array(starts, dtype=np.int64), size=int(max_windows_per_record), replace=False).tolist()
            starts.sort()
        idx.extend((r, int(s)) for s in starts)
    rng.shuffle(idx)
    return idx


def make_dataset_generator(
    items: list[tuple[RecordInfo, int]],
    *,
    fs_hz: float,
    win_sec: float,
    mode: str,
    envelope_rms_ms: float,
    downsample_hz: float,
    angle_stat: str,
    angle_max_deg: float,
    window_zscore: bool,
) -> Iterator[tuple[np.ndarray, float]]:
    win = int(round(win_sec * fs_hz))
    ds_factor = 1
    if mode == "envelope" and downsample_hz > 0:
        ds_factor = int(round(fs_hz / float(downsample_hz)))
        ds_factor = max(1, ds_factor)
    rms_win = int(round((envelope_rms_ms / 1000.0) * fs_hz))
    rms_win = max(1, rms_win)

    for r, s in items:
        arr = load_labeled_array(str(r.path))
        if s + win > arr.shape[0]:
            continue
        emg = arr[s : s + win, 0:2].astype(np.float32, copy=False)
        ang = arr[s : s + win, 2].astype(np.float32, copy=False)

        if mode == "envelope":
            emg = window_envelope(emg, rms_win=rms_win)
            emg = downsample(emg, factor=ds_factor)
        else:
            if ds_factor > 1:
                emg = downsample(emg, factor=ds_factor)

        if window_zscore:
            mu = np.mean(emg, axis=0, keepdims=True)
            sd = np.std(emg, axis=0, keepdims=True) + 1e-6
            emg = (emg - mu) / sd

        if angle_stat == "end":
            a_deg = float(ang[-1])
        else:
            a_deg = float(np.mean(ang))
        y = float(np.clip(a_deg / float(angle_max_deg), 0.0, 1.0))
        yield emg, y


def stack_dataset(
    items: list[tuple[RecordInfo, int]],
    *,
    fs_hz: float,
    win_sec: float,
    mode: str,
    envelope_rms_ms: float,
    downsample_hz: float,
    angle_stat: str,
    angle_max_deg: float,
    window_zscore: bool,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for emg, y in make_dataset_generator(
        items,
        fs_hz=fs_hz,
        win_sec=win_sec,
        mode=mode,
        envelope_rms_ms=envelope_rms_ms,
        downsample_hz=downsample_hz,
        angle_stat=angle_stat,
        angle_max_deg=angle_max_deg,
        window_zscore=window_zscore,
    ):
        xs.append(emg)
        ys.append(float(y))
    if not xs:
        return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    # Pad/crop to fixed length based on first sample
    input_len = xs[0].shape[0]
    fixed = []
    for x in xs:
        x = x[:input_len, :]
        if x.shape[0] < input_len:
            pad = input_len - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
        fixed.append(x.astype(np.float32))
    return np.stack(fixed, axis=0), np.array(ys, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute Conv1D EMG window dataset to NPZ.")
    p.add_argument("--input-root", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled")
    p.add_argument("--out-npz", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/conv1d_angle_dataset.npz")
    p.add_argument("--only-subject", type=str, default=None)
    p.add_argument("--split-by", choices=["record", "subject"], default="record")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fs", type=float, default=2000.0)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.05)
    p.add_argument("--max-windows-per-record", type=int, default=3000)

    p.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    p.add_argument("--envelope-rms-ms", type=float, default=50.0)
    p.add_argument("--downsample-hz", type=float, default=200.0)
    p.add_argument("--window-zscore", action="store_true")

    p.add_argument("--angle-stat", choices=["mean", "end"], default="mean")
    p.add_argument("--angle-max-deg", type=float, default=150.0)
    args = p.parse_args()

    input_root = Path(args.input_root)
    files = find_labeled_csvs(input_root)
    recs = build_record_index(files)
    recs = filter_only_subject(recs, args.only_subject)
    train_recs, val_recs, test_recs = split_records(
        recs,
        split_by=str(args.split_by),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    train_items = make_window_index(
        train_recs,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        step_sec=float(args.step_sec),
        max_windows_per_record=int(args.max_windows_per_record),
        seed=int(args.seed),
    )
    val_items = make_window_index(
        val_recs,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        step_sec=float(args.step_sec),
        max_windows_per_record=max(1, int(args.max_windows_per_record // 3)) if int(args.max_windows_per_record) > 0 else 0,
        seed=int(args.seed) + 1,
    )
    test_items = make_window_index(
        test_recs,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        step_sec=float(args.step_sec),
        max_windows_per_record=max(1, int(args.max_windows_per_record // 3)) if int(args.max_windows_per_record) > 0 else 0,
        seed=int(args.seed) + 2,
    )
    if not train_items or not val_items or not test_items:
        raise SystemExit("Split produced empty train/val/test window sets. Adjust --val-frac/--test-frac or dataset size.")

    x_train, y_train = stack_dataset(
        train_items,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        mode=str(args.mode),
        envelope_rms_ms=float(args.envelope_rms_ms),
        downsample_hz=float(args.downsample_hz),
        angle_stat=str(args.angle_stat),
        angle_max_deg=float(args.angle_max_deg),
        window_zscore=bool(args.window_zscore),
    )
    x_val, y_val = stack_dataset(
        val_items,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        mode=str(args.mode),
        envelope_rms_ms=float(args.envelope_rms_ms),
        downsample_hz=float(args.downsample_hz),
        angle_stat=str(args.angle_stat),
        angle_max_deg=float(args.angle_max_deg),
        window_zscore=bool(args.window_zscore),
    )
    x_test, y_test = stack_dataset(
        test_items,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        mode=str(args.mode),
        envelope_rms_ms=float(args.envelope_rms_ms),
        downsample_hz=float(args.downsample_hz),
        angle_stat=str(args.angle_stat),
        angle_max_deg=float(args.angle_max_deg),
        window_zscore=bool(args.window_zscore),
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        meta=json.dumps(
            {
                "input_root": str(input_root),
                "only_subject": str(args.only_subject) if args.only_subject else None,
                "split_by": str(args.split_by),
                "n_records": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
                "n_windows": {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
                "fs": float(args.fs),
                "win_sec": float(args.win_sec),
                "step_sec": float(args.step_sec),
                "mode": str(args.mode),
                "envelope_rms_ms": float(args.envelope_rms_ms),
                "downsample_hz": float(args.downsample_hz),
                "window_zscore": bool(args.window_zscore),
                "angle_stat": str(args.angle_stat),
                "angle_max_deg": float(args.angle_max_deg),
            },
            indent=2,
        ),
    )
    print(f"Saved dataset: {out_npz}")


if __name__ == "__main__":
    main()

