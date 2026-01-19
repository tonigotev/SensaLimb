from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_velocity_features_csv(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns:
      full_df: original df (for subject/record splits)
      xdf: feature columns only
      y: target array (float32), shape (N,)
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise SystemExit(f"features.csv must contain '{target_col}' (run label_mendeley_semg.py --target velocity).")

    y = df[target_col].to_numpy(dtype=np.float32, copy=False)

    drop_cols = [c for c in ["subject", "record", "t0_s", "t1_s", "label", "vel_dps", "vel_norm"] if c in df.columns]
    xdf = df.drop(columns=drop_cols)
    xdf = xdf.apply(pd.to_numeric, errors="coerce")

    # Drop any rows with NaNs in features or target
    mask = ~xdf.isna().any(axis=1) & ~pd.isna(df[target_col])
    xdf = xdf.loc[mask].reset_index(drop=True)
    y = y[mask.to_numpy()]
    full_df = df.loc[mask].reset_index(drop=True)

    return full_df, xdf, y


def split_rowwise(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)

def regression_bins(y: np.ndarray, *, n_bins: int, clip: tuple[float, float]) -> np.ndarray:
    lo, hi = float(clip[0]), float(clip[1])
    yy = np.clip(y.astype(np.float64, copy=False), lo, hi)
    t = (yy - lo) / max(hi - lo, 1e-12)
    b = np.floor(t * n_bins).astype(np.int64)
    return np.clip(b, 0, n_bins - 1)

def stratified_group_split_regression(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    group_col: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    n_bins: int,
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group split (record/subject) while trying to preserve target distribution.
    We bin y and then greedily assign groups to match global bin counts.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)

    ybin = regression_bins(y, n_bins=n_bins, clip=clip)
    total = np.bincount(ybin, minlength=n_bins).astype(np.float64)
    total_rows = int(np.sum(total))
    target_test = total * float(test_frac)
    target_val = total * float(val_frac)

    group_counts: dict[str, np.ndarray] = {}
    for g in uniq:
        mask = groups == g
        group_counts[str(g)] = np.bincount(ybin[mask], minlength=n_bins).astype(np.float64)

    group_list = list(group_counts.keys())
    group_list.sort(key=lambda k: float(np.sum(group_counts[k])), reverse=True)
    rng.shuffle(group_list)

    test_groups: list[str] = []
    val_groups: list[str] = []
    sum_test = np.zeros(n_bins, dtype=np.float64)
    sum_val = np.zeros(n_bins, dtype=np.float64)

    def score_add(current: np.ndarray, target: np.ndarray, add: np.ndarray) -> float:
        return float(np.sum(np.abs((current + add) - target)))

    for g in group_list:
        cnt = group_counts[g]

        test_ok = float(np.sum(sum_test)) < (test_frac * total_rows * 1.05)
        val_ok = float(np.sum(sum_val)) < (val_frac * total_rows * 1.05)

        candidates: list[tuple[float, str]] = []
        if test_ok:
            candidates.append((score_add(sum_test, target_test, cnt), "test"))
        if val_ok:
            candidates.append((score_add(sum_val, target_val, cnt), "val"))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            choice = candidates[0][1]
        else:
            choice = "train"

        if choice == "test":
            test_groups.append(g)
            sum_test += cnt
        elif choice == "val":
            val_groups.append(g)
            sum_val += cnt

    test_mask = np.isin(groups, np.array(test_groups, dtype=object))
    val_mask = np.isin(groups, np.array(val_groups, dtype=object)) & ~test_mask
    train_mask = ~(test_mask | val_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)

def make_history(
    full_df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    history: int,
    group_col: str = "record",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Concatenate features from previous (history-1) windows within each group.
    Requires columns: group_col and t0_s to sort windows.
    """
    if history <= 1:
        return full_df, x, y
    if group_col not in full_df.columns or "t0_s" not in full_df.columns:
        raise SystemExit(f"--history {history} requires features.csv to include '{group_col}' and 't0_s'")

    out_rows: list[int] = []
    out_x: list[np.ndarray] = []
    out_y: list[float] = []

    groups = full_df[group_col].astype(str).to_numpy()
    t0 = pd.to_numeric(full_df["t0_s"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(t0).any():
        raise SystemExit("features.csv has non-numeric t0_s; cannot build history features.")

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(t0[idx])]
        if idx.size < history:
            continue
        for i in range(history - 1, idx.size):
            window_idx = idx[i - (history - 1) : i + 1]
            stacked = x[window_idx].reshape(-1)
            out_rows.append(int(idx[i]))
            out_x.append(stacked)
            out_y.append(float(y[idx[i]]))

    if not out_rows:
        raise SystemExit(f"--history {history} produced zero samples (not enough windows per group).")

    keep = np.array(out_rows, dtype=np.int64)
    new_df = full_df.loc[keep].reset_index(drop=True)
    new_x = np.stack(out_x, axis=0).astype(np.float32)
    new_y = np.array(out_y, dtype=np.float32)
    return new_df, new_x, new_y


def group_split(df: pd.DataFrame, group_col: str, val_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    n = groups.shape[0]
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    test_groups: list[str] = []
    val_groups: list[str] = []

    def count_rows(gs: list[str]) -> int:
        if not gs:
            return 0
        return int(np.sum(np.isin(groups, np.array(gs, dtype=object))))

    for g in uniq:
        if count_rows(test_groups) < n_test:
            test_groups.append(str(g))
        elif count_rows(val_groups) < n_val:
            val_groups.append(str(g))
        else:
            break

    test_mask = np.isin(groups, np.array(test_groups, dtype=object))
    val_mask = np.isin(groups, np.array(val_groups, dtype=object)) & ~test_mask
    train_mask = ~(test_mask | val_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def plot_curves(history, out_path: Path) -> None:
    hist = history.history
    epochs = np.arange(1, len(next(iter(hist.values()))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist.get("loss", []), label="train")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, hist.get("mae", []), label="train")
    if "val_mae" in hist:
        axes[1].plot(epochs, hist["val_mae"], label="val")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    ax.plot([-1, 1], [-1, 1], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True (test)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP regressor for angle velocity and export TFLite.")
    parser.add_argument(
        "--features-csv",
        type=str,
        default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled/features.csv",
        help="features.csv produced by label_mendeley_semg.py --target velocity",
    )
    parser.add_argument("--target-col", choices=["vel_norm", "vel_dps"], default="vel_norm")
    parser.add_argument("--out-dir", type=str, default="ML/models/mlp_velocity", help="Output directory")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--split-by", choices=["row", "record", "subject"], default="row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--history", type=int, default=1, help="Number of past windows to concatenate (default: 1).")
    parser.add_argument("--bins", type=int, default=21, help="Regression stratification bins (default: 21).")
    parser.add_argument("--clip", type=str, default="-1,1", help='Clip range for stratification, e.g. "-1,1".')
    args = parser.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"TensorFlow is required. Import error: {exc}")

    features_csv = Path(args.features_csv)
    if not features_csv.is_file():
        raise SystemExit(f"features.csv not found: {features_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df, xdf, y = load_velocity_features_csv(features_csv, str(args.target_col))
    x = xdf.to_numpy(dtype=np.float32, copy=False)

    # Optional history stacking to provide temporal context (EMG -> motion has delay).
    full_df, x, y = make_history(full_df, x, y, history=int(args.history), group_col="record")

    val_frac = float(args.val_frac)
    test_frac = float(args.test_frac)
    if (val_frac + test_frac) >= 0.8:
        raise SystemExit("val+test fractions too large.")

    if str(args.split_by) == "row":
        train_idx, val_idx, test_idx = split_rowwise(x.shape[0], val_frac, test_frac, int(args.seed))
    else:
        group_col = str(args.split_by)
        if group_col not in full_df.columns:
            raise SystemExit(f"features.csv missing column '{group_col}' required for split-by {group_col}")
        clip_parts = [p.strip() for p in str(args.clip).split(",")]
        if len(clip_parts) != 2:
            raise SystemExit('Invalid --clip. Expected format like "-1,1"')
        clip = (float(clip_parts[0]), float(clip_parts[1]))

        # For regression targets, naive group split can produce extreme distribution shift.
        train_idx, val_idx, test_idx = stratified_group_split_regression(
            full_df,
            y,
            group_col=group_col,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=int(args.seed),
            n_bins=int(args.bins),
            clip=clip,
        )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(
        "Split sizes:",
        json.dumps({"train": int(y_train.size), "val": int(y_val.size), "test": int(y_test.size)}, indent=2),
    )

    # Normalization layer
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(x_train)

    hidden = [int(s.strip()) for s in str(args.hidden).split(",") if s.strip()]
    kernel_reg = tf.keras.regularizers.L2(float(args.l2)) if float(args.l2) > 0 else None

    layers: list[tf.keras.layers.Layer] = [
        tf.keras.Input(shape=(x.shape[1],), name="features"),
        norm,
    ]
    for i, h in enumerate(hidden):
        layers.append(tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=kernel_reg, name=f"dense_{i}_{h}"))
        if float(args.dropout) > 0:
            layers.append(tf.keras.layers.Dropout(float(args.dropout), name=f"dropout_{i}"))

    # Output in [-1, 1] for vel_norm, linear for vel_dps
    out_act = "tanh" if str(args.target_col) == "vel_norm" else "linear"
    layers.append(tf.keras.layers.Dense(1, activation=out_act, name="vel_out"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks: list[tf.keras.callbacks.Callback] = []
    best_path = out_dir / "best.keras"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True))
    if args.early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(args.patience), restore_best_weights=True))

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=2,
        callbacks=callbacks,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=int(args.batch_size), verbose=0).reshape(-1)
    corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if y_test.size > 2 else float("nan")

    curves_path = out_dir / "training_curves.png"
    plot_curves(history, curves_path)
    scatter_path = out_dir / "pred_vs_true.png"
    plot_pred_vs_true(y_test, y_pred, scatter_path)

    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.quantize_int8:
        def rep_data():
            n = min(2000, x_train.shape[0])
            for i in range(n):
                yield [x_train[i : i + 1]]

        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = out_dir / ("model_int8.tflite" if args.quantize_int8 else "model.tflite")
    tflite_path.write_bytes(tflite_model)

    meta = {
        "features_csv": str(features_csv),
        "target_col": str(args.target_col),
        "history": int(args.history),
        "n_features": int(x.shape[1]),
        "feature_columns": list(xdf.columns),
        "splits": {"split_by": str(args.split_by), "val_frac": val_frac, "test_frac": test_frac},
        "test_metrics": {"mse": float(test_loss), "mae": float(test_mae), "corr": corr},
        "training_curves_png": str(curves_path),
        "pred_vs_true_png": str(scatter_path),
        "tflite": str(tflite_path),
        "keras": str(keras_path),
        "quantized_int8": bool(args.quantize_int8),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {out_dir / 'meta.json'}")
    print(f"Saved: {tflite_path}")


if __name__ == "__main__":
    main()

