from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    subject: str
    record: str
    n_samples: int


def find_labeled_csvs(root: Path) -> list[Path]:
    # Expect structure like .../labeled/subject_1/*_labeled.csv
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
    # Fast-ish row count for large CSV without loading all columns.
    # For pandas, reading only first column is cheaper.
    try:
        return int(pd.read_csv(csv_path, header=None, usecols=[0]).shape[0])
    except Exception:
        # Fallback: read whole file
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
        # For raw-signal datasets we split by record anyway to avoid leakage of adjacent windows.
        split_by = "record"

    if split_by == "record":
        items = recs[:]
        rng.shuffle(items)
        n = len(items)
        # For small n, Python round() can yield 0 (banker's rounding). Ensure non-empty splits when possible.
        n_test = int(np.ceil(test_frac * n)) if test_frac > 0 else 0
        n_val = int(np.ceil(val_frac * n)) if val_frac > 0 else 0
        # Keep at least 1 in val/test if requested and feasible.
        if test_frac > 0 and n >= 3:
            n_test = max(1, n_test)
        if val_frac > 0 and n >= 3:
            n_val = max(1, n_val)
        # Ensure we don't exhaust all records
        if (n_test + n_val) >= n:
            # leave at least 1 train record
            overflow = (n_test + n_val) - (n - 1)
            # reduce val first, then test
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
    """
    Load labeled CSV (biceps,triceps,angle,label) without header.
    Returns float32 array shape (N, 4).
    """
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {csv_path}, got {arr.shape[1]}")
    # ensure 4 cols if label exists; if not, keep first 3
    if arr.shape[1] >= 4:
        arr = arr[:, :4]
    return arr


def window_envelope(x: np.ndarray, *, rms_win: int) -> np.ndarray:
    """
    Simple RMS envelope using a moving average of squared signal.
    """
    if rms_win <= 1:
        return np.abs(x)
    # pad reflect to reduce edge artifacts
    pad = rms_win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    sq = xp * xp
    # moving average via cumulative sum
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
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
            # raw mode: just optional downsample (not recommended)
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


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.25)
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True angle_norm")
    ax.set_ylabel("Predicted angle_norm")
    ax.set_title("Predicted vs True (test)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    p = argparse.ArgumentParser(description="Train a simple Conv1D+MLP regressor on EMG windows to predict angle.")
    p.add_argument("--input-root", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled")
    p.add_argument("--dataset-npz", type=str, default=None, help="Precomputed dataset (.npz) from build_conv1d_dataset.py")
    p.add_argument("--out-dir", type=str, default="ML/models/conv1d_angle")
    p.add_argument("--only-subject", type=str, default=None)
    p.add_argument("--split-by", choices=["record", "subject"], default="record")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fs", type=float, default=2000.0)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.05)
    p.add_argument("--max-windows-per-record", type=int, default=3000, help="0 = use all windows (can be huge).")

    p.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    p.add_argument("--envelope-rms-ms", type=float, default=50.0)
    p.add_argument("--downsample-hz", type=float, default=200.0, help="Only used for envelope/raw downsampling (0 disables).")
    p.add_argument("--window-zscore", action="store_true", help="Per-window z-score normalization of EMG channels.")

    p.add_argument("--angle-stat", choices=["mean", "end"], default="mean")
    p.add_argument("--angle-max-deg", type=float, default=150.0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffle buffer (faster startup, less randomness).")
    p.add_argument("--shuffle-buffer", type=int, default=20000, help="Shuffle buffer size (default: 20000).")
    p.add_argument("--steps-per-epoch", type=int, default=0, help="Limit training steps per epoch (0 = full).")
    args = p.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SystemExit(f"TensorFlow is required. Import error: {exc}")

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_npz:
        data = np.load(Path(args.dataset_npz), allow_pickle=True)
        x_train = data["x_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        x_val = data["x_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.float32)
        x_test = data["x_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        train_items = []
        val_items = []
        test_items = []
        fs_hz = float(args.fs)
        input_len = int(x_train.shape[1])
    else:
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
            raise SystemExit("Split produced empty train/val/test window sets. Adjust --val-frac/--test-frac or dataset size.")

        # Determine model input length after optional downsample/envelope
        fs_hz = float(args.fs)
        win = int(round(float(args.win_sec) * fs_hz))
        ds_factor = 1
        if float(args.downsample_hz) > 0:
            ds_factor = int(round(fs_hz / float(args.downsample_hz)))
            ds_factor = max(1, ds_factor)
        input_len = win if str(args.mode) == "raw" and ds_factor <= 1 else int(max(1, round(win / ds_factor)))

    out_act = "sigmoid"
    model = build_conv1d_model(input_len=input_len, n_ch=2, out_act=out_act, dropout=float(args.dropout))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    def make_tf_dataset(items: list[tuple[RecordInfo, int]], *, shuffle: bool) -> "tf.data.Dataset":
        # Use from_generator to avoid holding huge arrays in RAM.
        # Yield variable-length windows depending on mode; we pad/crop to a fixed length.
        if args.dataset_npz:
            # use precomputed arrays
            if items == "train":
                ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            elif items == "val":
                ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            else:
                ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            # ensure shapes
            ds = ds.map(lambda x, y: (tf.ensure_shape(x, (input_len, 2)), tf.ensure_shape(tf.reshape(y, (1,)), (1,))), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            gen = lambda: make_dataset_generator(
                items,
                fs_hz=float(args.fs),
                win_sec=float(args.win_sec),
                mode=str(args.mode),
                envelope_rms_ms=float(args.envelope_rms_ms),
                downsample_hz=float(args.downsample_hz),
                angle_stat=str(args.angle_stat),
                angle_max_deg=float(args.angle_max_deg),
                window_zscore=bool(args.window_zscore),
            )
            output_signature = (
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            )
            ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

            def fix_len(x, y):
                x = x[:input_len, :]
                pad = tf.maximum(0, input_len - tf.shape(x)[0])
                x = tf.pad(x, [[0, pad], [0, 0]])
                x.set_shape((input_len, 2))
                y.set_shape((1,))
                return x, y

            ds = ds.map(fix_len, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            buf = int(args.shuffle_buffer)
            if buf <= 0:
                buf = min(20_000, max(2_000, len(items)))
            ds = ds.shuffle(buffer_size=min(buf, max(2_000, len(items))), reshuffle_each_iteration=True)
        ds = ds.batch(int(args.batch_size)).prefetch(tf.data.AUTOTUNE)
        return ds

    if args.dataset_npz:
        train_ds = make_tf_dataset("train", shuffle=not bool(args.no_shuffle))
        val_ds = make_tf_dataset("val", shuffle=False)
        test_ds = make_tf_dataset("test", shuffle=False)
    else:
        train_ds = make_tf_dataset(train_items, shuffle=not bool(args.no_shuffle))
        val_ds = make_tf_dataset(val_items, shuffle=False)
        test_ds = make_tf_dataset(test_items, shuffle=False)

    # Baseline mean predictor (approx from a subset to keep it cheap)
    if args.dataset_npz:
        y_train_mean = float(np.mean(y_train)) if y_train.size else 0.5
        print("Split windows:", json.dumps({"train": int(y_train.shape[0]), "val": int(y_val.shape[0]), "test": int(y_test.shape[0])}, indent=2))
    else:
        y_baseline_samples: list[float] = []
        for _, yy in make_dataset_generator(
            train_items[:2000],
            fs_hz=fs_hz,
            win_sec=float(args.win_sec),
            mode=str(args.mode),
            envelope_rms_ms=float(args.envelope_rms_ms),
            downsample_hz=float(args.downsample_hz),
            angle_stat=str(args.angle_stat),
            angle_max_deg=float(args.angle_max_deg),
            window_zscore=bool(args.window_zscore),
        ):
            y_baseline_samples.append(float(yy.reshape(-1)[0]))
        y_train_mean = float(np.mean(y_baseline_samples)) if y_baseline_samples else 0.5
        print("Split windows:", json.dumps({"train": len(train_items), "val": len(val_items), "test": len(test_items)}, indent=2))
    print("Baseline mean (approx, from subset):", y_train_mean)

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

    # Collect predictions for scatter
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
    plot_pred_vs_true(yt, yp, scatter_path)

    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)

    if args.dataset_npz:
        n_records_meta = None
        n_windows_meta = {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])}
    else:
        n_records_meta = {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)}
        n_windows_meta = {"train": len(train_items), "val": len(val_items), "test": len(test_items)}

    meta = {
        "input_root": str(input_root),
        "mode": str(args.mode),
        "fs": float(args.fs),
        "win_sec": float(args.win_sec),
        "step_sec": float(args.step_sec),
        "downsample_hz": float(args.downsample_hz),
        "envelope_rms_ms": float(args.envelope_rms_ms),
        "window_zscore": bool(args.window_zscore),
        "angle_stat": str(args.angle_stat),
        "angle_max_deg": float(args.angle_max_deg),
        "split_by": str(args.split_by),
        "only_subject": str(args.only_subject) if args.only_subject else None,
        "n_records": n_records_meta,
        "n_windows": n_windows_meta,
        "baseline_mean_subset": y_train_mean,
        "test_metrics": {"loss": float(test_loss), "mae": float(test_mae), "corr": corr},
        "training_curves_png": str(curves_path),
        "pred_vs_true_png": str(scatter_path),
        "keras": str(keras_path),
        "tflite": str(tflite_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

