from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    subject: str
    record: str
    n_samples: int
    split: str | None
    movement: str | None
    load_g: int | None


def normalize_subject_name(folder: str, file_stem: str) -> str:
    low = folder.strip().casefold()
    if low.startswith("subject"):
        digits = "".join(ch for ch in folder if ch.isdigit())
        if digits:
            return f"subject_{digits}"
    # fallback to filename prefix
    subj = file_stem.split("_", 1)[0]
    if subj.isdigit():
        return f"subject_{subj}"
    return folder.strip().replace(" ", "_")


def parse_file_name(stem: str) -> tuple[str | None, str | None, int | None]:
    # Expected: "<id>_<movement>_<train|test>_<load>"
    parts = stem.split("_")
    if len(parts) < 4:
        return None, None, None
    movement = parts[1]
    split = parts[2]
    load_g = None
    if parts[3].isdigit():
        load_g = int(parts[3])
    return split, movement, load_g


def find_toro_files(root: Path) -> list[RecordInfo]:
    if not root.exists():
        raise SystemExit(f"Input root not found: {root}")
    files = sorted([p for p in root.rglob("*.txt") if p.is_file() and p.name != "subject_info.txt"])
    if not files:
        raise SystemExit(f"No .txt files found under: {root}")
    out: list[RecordInfo] = []
    for p in files:
        split, movement, load_g = parse_file_name(p.stem)
        subject = normalize_subject_name(p.parent.name, p.stem)
        n = count_rows_fast(p)
        out.append(RecordInfo(path=p, subject=subject, record=p.stem, n_samples=n, split=split, movement=movement, load_g=load_g))
    return out


def count_rows_fast(path: Path) -> int:
    try:
        return int(pd.read_csv(path, header=None, usecols=[0]).shape[0])
    except Exception:
        return int(pd.read_csv(path, header=None).shape[0])


def filter_recs(
    recs: list[RecordInfo],
    *,
    only_subject: str | None,
    movement: str,
    load: str,
) -> list[RecordInfo]:
    kept = recs[:]
    if only_subject:
        key = str(only_subject).casefold()
        kept = [r for r in kept if r.subject.casefold() == key or r.subject.replace("subject_", "subject ").casefold() == key]
        if not kept:
            raise SystemExit(f"No records match --only-subject {only_subject}. Example subjects: {sorted({r.subject for r in recs})[:10]}")
    if movement != "all":
        kept = [r for r in kept if r.movement == movement]
    if load != "all":
        kept = [r for r in kept if r.load_g is not None and str(r.load_g) == load]
    if not kept:
        raise SystemExit("No records left after filtering by subject/movement/load.")
    return kept


def split_records_by_file(
    recs: list[RecordInfo],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[RecordInfo], list[RecordInfo], list[RecordInfo]]:
    train_all = [r for r in recs if r.split == "train"]
    test = [r for r in recs if r.split == "test"]
    if not train_all or not test:
        raise SystemExit("File-based split requested, but train/test files were not found.")
    rng = np.random.default_rng(seed)
    items = train_all[:]
    rng.shuffle(items)
    n = len(items)
    n_val = int(np.ceil(val_frac * n)) if val_frac > 0 else 0
    if val_frac > 0 and n >= 3:
        n_val = max(1, n_val)
    if n_val >= n:
        n_val = max(0, n - 1)
    val = items[:n_val]
    train = items[n_val:]
    return train, val, test


def split_records_generic(
    recs: list[RecordInfo],
    *,
    split_by: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[RecordInfo], list[RecordInfo], list[RecordInfo]]:
    rng = np.random.default_rng(seed)
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


@lru_cache(maxsize=16)
def load_toro_array(txt_path: str) -> np.ndarray:
    df = pd.read_csv(txt_path, header=None, sep=r"\s+")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {txt_path}, got {arr.shape[1]}")
    return arr[:, :5]


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


def compute_angle_range(recs: list[RecordInfo]) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    for r in recs:
        arr = load_toro_array(str(r.path))
        a = arr[:, 4]
        if a.size == 0:
            continue
        lo = min(lo, float(np.min(a)))
        hi = max(hi, float(np.max(a)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise SystemExit("Failed to compute angle range from dataset.")
    if hi <= lo:
        raise SystemExit("Angle range is degenerate; min >= max.")
    return lo, hi


def make_dataset_generator(
    items: list[tuple[RecordInfo, int]],
    *,
    fs_hz: float,
    win_sec: float,
    mode: str,
    envelope_rms_ms: float,
    downsample_hz: float,
    window_zscore: bool,
    emg_source: str,
    angle_norm: str,
    angle_min: float | None,
    angle_max: float | None,
    angle_history_sec: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    win = int(round(win_sec * fs_hz))
    ds_factor = 1
    if mode == "envelope" and downsample_hz > 0:
        ds_factor = int(round(fs_hz / float(downsample_hz)))
        ds_factor = max(1, ds_factor)
    rms_win = int(round((envelope_rms_ms / 1000.0) * fs_hz))
    rms_win = max(1, rms_win)
    lag = int(round(angle_history_sec * fs_hz))

    for r, s in items:
        arr = load_toro_array(str(r.path))
        if s + win > arr.shape[0]:
            continue
        if emg_source == "filtered":
            emg = arr[s : s + win, 2:4].astype(np.float32, copy=False)
        else:
            emg = arr[s : s + win, 0:2].astype(np.float32, copy=False)
        ang = arr[s : s + win, 4].astype(np.float32, copy=False)
        hist = None
        if angle_history_sec > 0:
            if s - lag < 0 or (s - lag + win) > arr.shape[0]:
                continue
            hist = arr[s - lag : s - lag + win, 4].astype(np.float32, copy=False)

        if mode == "envelope":
            emg = window_envelope(emg, rms_win=rms_win)
            emg = downsample(emg, factor=ds_factor)
        else:
            if ds_factor > 1:
                emg = downsample(emg, factor=ds_factor)
        if hist is not None and ds_factor > 1:
            hist = downsample(hist.reshape(-1, 1), factor=ds_factor).reshape(-1)

        if window_zscore:
            mu = np.mean(emg, axis=0, keepdims=True)
            sd = np.std(emg, axis=0, keepdims=True) + 1e-6
            emg = (emg - mu) / sd

        a_deg = float(np.mean(ang))
        if angle_norm == "none":
            y = a_deg
        else:
            if angle_min is None or angle_max is None:
                raise SystemExit("Angle min/max must be provided for minmax normalization.")
            denom = float(angle_max - angle_min)
            if denom <= 0:
                y = 0.5
            else:
                y = (a_deg - float(angle_min)) / denom
                y = float(np.clip(y, 0.0, 1.0))
        if hist is not None:
            if angle_norm == "none":
                hist_norm = hist
            else:
                denom = float(angle_max - angle_min) if angle_min is not None and angle_max is not None else 1.0
                if denom <= 0:
                    hist_norm = np.zeros_like(hist)
                else:
                    hist_norm = (hist - float(angle_min)) / denom
                    hist_norm = np.clip(hist_norm, 0.0, 1.0)
            emg = np.concatenate([emg, hist_norm.reshape(-1, 1)], axis=1)
        yield emg, np.array([y], dtype=np.float32)


def plot_curves(history, out_path: Path) -> None:
    hist = history.history
    epochs = np.arange(1, len(next(iter(hist.values()))) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist.get("loss", []), label="train")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, hist.get("mae", []), label="train")
    if "val_mae" in hist:
        axes[1].plot(epochs, hist["val_mae"], label="val")
    axes[1].set_title("MAE")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, *, angle_norm: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.25)
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    label = "angle_norm" if angle_norm != "none" else "angle_deg"
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title("Predicted vs True (test)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_angle_histograms(
    hist_data: dict[str, np.ndarray],
    out_path: Path,
    *,
    angle_norm: str,
    n_bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    label = "angle_norm" if angle_norm != "none" else "angle_deg"
    for name, values in hist_data.items():
        if values.size == 0:
            continue
        ax.hist(values, bins=n_bins, alpha=0.45, density=True, label=name)
    ax.set_title(f"{label} distribution (sampled)")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_angles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def build_conv1d_model(*, input_len: int, n_ch: int, out_act: str, dropout: float):
    import tensorflow as tf  # type: ignore

    inp = tf.keras.Input(shape=(input_len, n_ch), name="emg")
    x = tf.keras.layers.Conv1D(32, 9, strides=2, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(64, 7, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(128, 5, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation=out_act, name="angle_out")(x)
    return tf.keras.Model(inp, out)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Conv1D regressor on Toro Ossaba sEMG+angle dataset.")
    p.add_argument("--input-root", type=str, default="ML/datasets/Toro Ossaba")
    p.add_argument("--out-dir", type=str, default="ML/models/conv1d_angle_toro_ossaba")
    p.add_argument("--run-name", type=str, default=None, help="Optional subfolder name for this run.")
    p.add_argument("--auto-run-name", action="store_true", help="Create a timestamped subfolder for this run.")
    p.add_argument("--only-subject", type=str, default=None)
    p.add_argument("--movement", choices=["flex", "pronsup", "all"], default="flex")
    p.add_argument("--load", choices=["0", "1360", "2270", "all"], default="all")
    p.add_argument("--use-file-split", action="store_true", help="Use dataset train/test files. Val is a subset of train.")
    p.add_argument("--split-by", choices=["record", "subject"], default="record", help="Used only when --use-file-split is not set.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fs", type=float, default=1024.0)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.05)
    p.add_argument("--max-windows-per-record", type=int, default=3000, help="0 = use all windows (can be huge).")

    p.add_argument("--emg-source", choices=["filtered", "raw"], default="filtered")
    p.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    p.add_argument("--envelope-rms-ms", type=float, default=50.0)
    p.add_argument("--downsample-hz", type=float, default=200.0, help="Only used for envelope/raw downsampling (0 disables).")
    p.add_argument("--window-zscore", action="store_true", help="Per-window z-score normalization of EMG channels.")

    p.add_argument("--angle-norm", choices=["minmax", "none"], default="minmax")
    p.add_argument("--angle-min", type=float, default=None)
    p.add_argument("--angle-max", type=float, default=None)
    p.add_argument("--angle-history-sec", type=float, default=0.0, help="Include past angle channel (lagged by this many seconds).")
    p.add_argument("--histogram-samples", type=int, default=20000, help="Max windows per split to summarize histograms.")
    p.add_argument("--histogram-bins", type=int, default=40, help="Histogram bin count.")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffle buffer (faster startup, less randomness).")
    p.add_argument("--shuffle-buffer", type=int, default=20000, help="Shuffle buffer size (default: 20000).")
    p.add_argument("--steps-per-epoch", type=int, default=0, help="Limit training steps per epoch (0 = full).")
    p.add_argument("--cache", choices=["none", "memory", "disk"], default="none", help="Cache dataset after preprocessing.")
    p.add_argument("--cache-path", type=str, default=None, help="Optional cache directory (used when --cache=disk).")
    p.add_argument("--repeat", action="store_true", help="Repeat training dataset to avoid running out of data.")
    p.add_argument("--tflite-int8", action="store_true", help="Export int8 quantized TFLite model (requires representative data).")
    p.add_argument("--tflite-rep-samples", type=int, default=500, help="Representative samples for int8 quantization.")
    args = p.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SystemExit(f"TensorFlow is required. Import error: {exc}")

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    if args.run_name or bool(args.auto_run_name):
        if args.run_name:
            run_name = str(args.run_name)
        else:
            from datetime import datetime

            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = find_toro_files(input_root)
    recs = filter_recs(
        recs,
        only_subject=args.only_subject,
        movement=str(args.movement),
        load=str(args.load),
    )

    if bool(args.use_file_split):
        train_recs, val_recs, test_recs = split_records_by_file(
            recs,
            val_frac=float(args.val_frac),
            seed=int(args.seed),
        )
        split_method = "file"
    else:
        train_recs, val_recs, test_recs = split_records_generic(
            recs,
            split_by=str(args.split_by),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
            seed=int(args.seed),
        )
        split_method = str(args.split_by)

    # Determine angle normalization range from training set if needed
    angle_min = args.angle_min
    angle_max = args.angle_max
    if str(args.angle_norm) == "minmax":
        if angle_min is None or angle_max is None:
            angle_min, angle_max = compute_angle_range(train_recs)

    # Build window indices
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
        raise SystemExit("Split produced empty train/val/test window sets. Adjust --val-frac/--test-frac or dataset filters.")

    fs_hz = float(args.fs)
    win = int(round(float(args.win_sec) * fs_hz))
    ds_factor = 1
    if float(args.downsample_hz) > 0:
        ds_factor = int(round(fs_hz / float(args.downsample_hz)))
        ds_factor = max(1, ds_factor)
    input_len = win if str(args.mode) == "raw" and ds_factor <= 1 else int(max(1, round(win / ds_factor)))
    n_ch = 3 if float(args.angle_history_sec) > 0 else 2

    out_act = "linear" if str(args.angle_norm) == "none" else "sigmoid"
    model = build_conv1d_model(input_len=input_len, n_ch=n_ch, out_act=out_act, dropout=float(args.dropout))
    loss = tf.keras.losses.Huber(delta=0.05) if str(args.angle_norm) != "none" else tf.keras.losses.MeanAbsoluteError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    def make_tf_dataset(
        items: list[tuple[RecordInfo, int]],
        *,
        split_name: str,
        shuffle: bool,
        repeat: bool,
    ) -> "tf.data.Dataset":
        gen = lambda: make_dataset_generator(
            items,
            fs_hz=float(args.fs),
            win_sec=float(args.win_sec),
            mode=str(args.mode),
            envelope_rms_ms=float(args.envelope_rms_ms),
            downsample_hz=float(args.downsample_hz),
            window_zscore=bool(args.window_zscore),
            emg_source=str(args.emg_source),
            angle_norm=str(args.angle_norm),
            angle_min=angle_min,
            angle_max=angle_max,
            angle_history_sec=float(args.angle_history_sec),
        )
        output_signature = (
            tf.TensorSpec(shape=(None, n_ch), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        def fix_len(x, y):
            x = x[:input_len, :]
            pad = tf.maximum(0, input_len - tf.shape(x)[0])
            x = tf.pad(x, [[0, pad], [0, 0]])
            x.set_shape((input_len, n_ch))
            y.set_shape((1,))
            return x, y

        ds = ds.map(fix_len, num_parallel_calls=tf.data.AUTOTUNE)
        if str(args.cache) != "none":
            if str(args.cache) == "memory":
                ds = ds.cache()
            else:
                cache_root = Path(args.cache_path) if args.cache_path else (out_dir / "cache")
                cache_root.mkdir(parents=True, exist_ok=True)
                cache_file = cache_root / f"{split_name}.cache"
                ds = ds.cache(str(cache_file))
        if shuffle:
            buf = int(args.shuffle_buffer)
            if buf <= 0:
                buf = min(20_000, max(2_000, len(items)))
            ds = ds.shuffle(buffer_size=min(buf, max(2_000, len(items))), reshuffle_each_iteration=True)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(int(args.batch_size)).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_tf_dataset(train_items, split_name="train", shuffle=not bool(args.no_shuffle), repeat=bool(args.repeat))
    val_ds = make_tf_dataset(val_items, split_name="val", shuffle=False, repeat=False)
    test_ds = make_tf_dataset(test_items, split_name="test", shuffle=False, repeat=False)

    # Baseline mean predictor (approx from a subset to keep it cheap)
    y_baseline_samples: list[float] = []
    for _, yy in make_dataset_generator(
        train_items[:2000],
        fs_hz=fs_hz,
        win_sec=float(args.win_sec),
        mode=str(args.mode),
        envelope_rms_ms=float(args.envelope_rms_ms),
        downsample_hz=float(args.downsample_hz),
        window_zscore=bool(args.window_zscore),
        emg_source=str(args.emg_source),
        angle_norm=str(args.angle_norm),
        angle_min=angle_min,
        angle_max=angle_max,
        angle_history_sec=float(args.angle_history_sec),
    ):
        y_baseline_samples.append(float(yy.reshape(-1)[0]))
    y_train_mean = float(np.mean(y_baseline_samples)) if y_baseline_samples else 0.5
    print("Split windows:", json.dumps({"train": len(train_items), "val": len(val_items), "test": len(test_items)}, indent=2))
    print("Baseline mean (approx, from subset):", y_train_mean)

    def collect_targets(items: list[tuple[RecordInfo, int]], *, max_samples: int) -> np.ndarray:
        y_vals: list[float] = []
        for _, yy in make_dataset_generator(
            items[:max_samples],
            fs_hz=fs_hz,
            win_sec=float(args.win_sec),
            mode=str(args.mode),
            envelope_rms_ms=float(args.envelope_rms_ms),
            downsample_hz=float(args.downsample_hz),
            window_zscore=bool(args.window_zscore),
            emg_source=str(args.emg_source),
            angle_norm=str(args.angle_norm),
            angle_min=angle_min,
            angle_max=angle_max,
            angle_history_sec=float(args.angle_history_sec),
        ):
            y_vals.append(float(yy.reshape(-1)[0]))
        return np.array(y_vals, dtype=np.float32)

    hist_max = int(args.histogram_samples)
    train_y_sample = collect_targets(train_items, max_samples=hist_max)
    val_y_sample = collect_targets(val_items, max_samples=hist_max)
    test_y_sample = collect_targets(test_items, max_samples=hist_max)

    hist_json = {
        "train": summarize_angles(train_y_sample),
        "val": summarize_angles(val_y_sample),
        "test": summarize_angles(test_y_sample),
    }
    hist_data = {"train": train_y_sample, "val": val_y_sample, "test": test_y_sample}
    hist_png = out_dir / "angle_histograms.png"
    plot_angle_histograms(hist_data, hist_png, angle_norm=str(args.angle_norm), n_bins=int(args.histogram_bins))
    hist_json_path = out_dir / "angle_histograms.json"
    hist_json_path.write_text(json.dumps(hist_json, indent=2), encoding="utf-8")

    callbacks: list[tf.keras.callbacks.Callback] = []
    best_path = out_dir / "best.keras"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True))
    if bool(args.early_stop):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(args.patience), restore_best_weights=True))

    steps_per_epoch = int(args.steps_per_epoch) if int(args.steps_per_epoch) > 0 else None
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(args.epochs),
        verbose=2,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
    )

    test_loss, test_mae = model.evaluate(test_ds, verbose=0)

    y_true: list[float] = []
    y_pred: list[float] = []
    for xb, yb in test_ds:
        p = model.predict(xb, verbose=0).reshape(-1)
        y_pred.extend(p.tolist())
        y_true.extend(yb.numpy().reshape(-1).tolist())
    yt = np.array(y_true, dtype=np.float32)
    yp = np.array(y_pred, dtype=np.float32)
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 2 else float("nan")

    curves_path = out_dir / "training_curves.png"
    plot_curves(history, curves_path)
    scatter_path = out_dir / "pred_vs_true.png"
    plot_pred_vs_true(yt, yp, scatter_path, angle_norm=str(args.angle_norm))

    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)

    int8_path = None
    if bool(args.tflite_int8):
        rep_max = int(args.tflite_rep_samples)
        if rep_max <= 0:
            rep_max = 200

        def rep_data_gen():
            count = 0
            for x, _ in make_dataset_generator(
                train_items[: max(2000, rep_max * 2)],
                fs_hz=fs_hz,
                win_sec=float(args.win_sec),
                mode=str(args.mode),
                envelope_rms_ms=float(args.envelope_rms_ms),
                downsample_hz=float(args.downsample_hz),
                window_zscore=bool(args.window_zscore),
                emg_source=str(args.emg_source),
                angle_norm=str(args.angle_norm),
                angle_min=angle_min,
                angle_max=angle_max,
                angle_history_sec=float(args.angle_history_sec),
            ):
                x = x[:input_len, :]
                if x.shape[0] < input_len:
                    pad = np.zeros((input_len - x.shape[0], x.shape[1]), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=0)
                x = x.astype(np.float32, copy=False)
                yield [x.reshape(1, input_len, n_ch)]
                count += 1
                if count >= rep_max:
                    break

        int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int8_converter.representative_dataset = rep_data_gen
        int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int8_converter.inference_input_type = tf.int8
        int8_converter.inference_output_type = tf.int8
        tflite_int8 = int8_converter.convert()
        int8_path = out_dir / "model_int8.tflite"
        int8_path.write_bytes(tflite_int8)

    meta = {
        "input_root": str(input_root),
        "emg_source": str(args.emg_source),
        "mode": str(args.mode),
        "fs": float(args.fs),
        "win_sec": float(args.win_sec),
        "step_sec": float(args.step_sec),
        "downsample_hz": float(args.downsample_hz),
        "envelope_rms_ms": float(args.envelope_rms_ms),
        "window_zscore": bool(args.window_zscore),
        "angle_norm": str(args.angle_norm),
        "angle_min": float(angle_min) if angle_min is not None else None,
        "angle_max": float(angle_max) if angle_max is not None else None,
        "angle_history_sec": float(args.angle_history_sec),
        "split_method": split_method,
        "only_subject": str(args.only_subject) if args.only_subject else None,
        "movement": str(args.movement),
        "load": str(args.load),
        "n_records": {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)},
        "n_windows": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
        "baseline_mean_subset": y_train_mean,
        "angle_histograms_png": str(hist_png),
        "angle_histograms_json": str(hist_json_path),
        "angle_histogram_samples": int(args.histogram_samples),
        "test_metrics": {"loss": float(test_loss), "mae": float(test_mae), "corr": corr},
        "training_curves_png": str(curves_path),
        "pred_vs_true_png": str(scatter_path),
        "keras": str(keras_path),
        "tflite": str(tflite_path),
        "tflite_int8": str(int8_path) if int8_path else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
