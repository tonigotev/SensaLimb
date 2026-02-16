from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_features(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise SystemExit(f"features.csv must contain '{target_col}'. (Did you run label_mendeley_semg.py --target angle?)")

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    # Drop metadata + all known targets (never leak target into inputs).
    drop_cols = [c for c in ["subject", "record", "t0_s", "t1_s", "label", "vel_dps", "vel_norm", "angle_deg", "angle_norm"] if c in df.columns]
    xdf = df.drop(columns=drop_cols)
    if target_col in xdf.columns:
        xdf = xdf.drop(columns=[target_col])
    xdf = xdf.apply(pd.to_numeric, errors="coerce")

    mask = ~xdf.isna().any(axis=1) & ~pd.isna(df[target_col])
    full_df = df.loc[mask].reset_index(drop=True)
    xdf = xdf.loc[mask].reset_index(drop=True)
    y = y[mask.to_numpy()]
    return full_df, xdf, y


def filter_only_subject(full_df: pd.DataFrame, x: np.ndarray, y: np.ndarray, subject: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if "subject" not in full_df.columns:
        raise SystemExit("--only-subject requires features.csv to include a 'subject' column.")
    subj = str(subject).strip()
    if not subj:
        return full_df, x, y
    s = full_df["subject"].astype(str).str.casefold()
    mask = (s == subj.casefold()).to_numpy()
    if not bool(np.any(mask)):
        uniq = sorted(set(full_df["subject"].astype(str).tolist()))
        raise SystemExit(f"No rows for --only-subject '{subject}'. Available subjects: {uniq}")
    return full_df.loc[mask].reset_index(drop=True), x[mask], y[mask]


def make_sequences(
    full_df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seq_len: int,
    group_col: str = "record",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build sequences of length seq_len within each group (record), ordered by t0_s.
    Output:
      new_df: rows corresponding to the LAST element of each sequence
      xs: (N, seq_len, F)
      ys: (N,)
    """
    if seq_len <= 1:
        raise SystemExit("--seq-len must be >= 2 for TCN.")
    if group_col not in full_df.columns or "t0_s" not in full_df.columns:
        raise SystemExit(f"--seq-len requires features.csv to include '{group_col}' and 't0_s'.")

    groups = full_df[group_col].astype(str).to_numpy()
    t0 = pd.to_numeric(full_df["t0_s"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(t0).any():
        raise SystemExit("features.csv has non-numeric t0_s; cannot build sequences.")

    out_rows: list[int] = []
    out_x: list[np.ndarray] = []
    out_y: list[float] = []

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(t0[idx])]
        if idx.size < seq_len:
            continue
        for i in range(seq_len - 1, idx.size):
            widx = idx[i - (seq_len - 1) : i + 1]
            out_rows.append(int(idx[i]))
            out_x.append(x[widx])
            out_y.append(float(y[idx[i]]))

    if not out_rows:
        raise SystemExit(f"--seq-len {seq_len} produced zero samples.")

    keep = np.array(out_rows, dtype=np.int64)
    new_df = full_df.loc[keep].reset_index(drop=True)
    xs = np.stack(out_x, axis=0).astype(np.float32)  # (N, L, F)
    ys = np.array(out_y, dtype=np.float32)
    return new_df, xs, ys


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
        m = groups == g
        group_counts[str(g)] = np.bincount(ybin[m], minlength=n_bins).astype(np.float64)

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

    train_idx = np.where(train_mask)[0].astype(np.int64)
    val_idx = np.where(val_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


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
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True (test)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mae_by_bins(y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int = 20) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (y_true >= lo) & (y_true < hi if i < n_bins - 1 else y_true <= hi)
        if not np.any(m):
            continue
        mae = float(np.mean(np.abs(y_true[m] - y_pred[m])))
        out.append({"bin_lo": lo, "bin_hi": hi, "n": float(np.sum(m)), "mae": mae})
    return out


def build_tcn_model(*, seq_len: int, n_feat: int, out_act: str, n_blocks: int, channels: int, kernel: int, dropout: float):
    import tensorflow as tf  # type: ignore

    inp = tf.keras.Input(shape=(seq_len, n_feat), name="seq_features")
    x = tf.keras.layers.LayerNormalization(axis=-1, name="ln_in")(inp)

    def res_block(x0, dilation: int, i: int):
        x = tf.keras.layers.Conv1D(channels, kernel_size=kernel, dilation_rate=dilation, padding="causal", name=f"c{i}_1_d{dilation}")(x0)
        x = tf.keras.layers.Activation("relu", name=f"relu{i}_1")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"drop{i}_1")(x)
        x = tf.keras.layers.Conv1D(channels, kernel_size=kernel, dilation_rate=dilation, padding="causal", name=f"c{i}_2_d{dilation}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu{i}_2")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"drop{i}_2")(x)

        if x0.shape[-1] != channels:
            skip = tf.keras.layers.Conv1D(channels, kernel_size=1, padding="same", name=f"skip{i}")(x0)
        else:
            skip = x0
        return tf.keras.layers.Add(name=f"add{i}")([skip, x])

    dilations = [2**i for i in range(n_blocks)]
    for i, d in enumerate(dilations):
        x = res_block(x, dilation=int(d), i=i)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x)
    out = tf.keras.layers.Dense(1, activation=out_act, name="angle_out")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a small TCN on sequences of window features to predict angle.")
    p.add_argument("--features-csv", type=str, default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled_angle/features.csv")
    p.add_argument("--target-col", choices=["angle_norm", "angle_deg"], default="angle_norm")
    p.add_argument("--out-dir", type=str, default="ML/models/tcn_angle")
    p.add_argument("--split-by", choices=["row", "record", "subject"], default="record")
    p.add_argument("--only-subject", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--bins", type=int, default=21)
    p.add_argument("--clip", type=str, default="0,1")

    # TCN hyperparams
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optional per-record calibration (fine-tune last layer)
    p.add_argument("--calibrate-per-record", action="store_true")
    p.add_argument("--calib-windows", type=int, default=200)
    p.add_argument("--calib-epochs", type=int, default=5)
    p.add_argument("--calib-lr", type=float, default=1e-4)
    args = p.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SystemExit(f"TensorFlow is required. Import error: {exc}")

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df, xdf, y = load_features(features_csv, str(args.target_col))
    x = xdf.to_numpy(dtype=np.float32, copy=False)
    if args.only_subject:
        full_df, x, y = filter_only_subject(full_df, x, y, str(args.only_subject))

    full_df, xs, ys = make_sequences(full_df, x, y, seq_len=int(args.seq_len), group_col="record")

    # split
    val_frac = float(args.val_frac)
    test_frac = float(args.test_frac)
    if (val_frac + test_frac) >= 0.8:
        raise SystemExit("val+test fractions too large.")

    if str(args.split_by) == "row":
        train_idx, val_idx, test_idx = split_rowwise(xs.shape[0], val_frac, test_frac, int(args.seed))
    else:
        group_col = str(args.split_by)
        if group_col not in full_df.columns:
            raise SystemExit(f"Missing column '{group_col}' required for --split-by {group_col}")
        clip_parts = [p.strip() for p in str(args.clip).split(",")]
        if len(clip_parts) != 2:
            raise SystemExit('Invalid --clip. Expected format like "0,1"')
        clip = (float(clip_parts[0]), float(clip_parts[1]))
        train_idx, val_idx, test_idx = stratified_group_split_regression(
            full_df,
            ys,
            group_col=group_col,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=int(args.seed),
            n_bins=int(args.bins),
            clip=clip,
        )

    x_train, y_train = xs[train_idx], ys[train_idx]
    x_val, y_val = xs[val_idx], ys[val_idx]
    x_test, y_test = xs[test_idx], ys[test_idx]

    # Baseline: mean predictor
    baseline = float(np.mean(y_train)) if y_train.size else 0.5
    baseline_mae = float(np.mean(np.abs(y_test - baseline))) if y_test.size else float("nan")
    print("Split sizes:", json.dumps({"train": int(y_train.size), "val": int(y_val.size), "test": int(y_test.size)}, indent=2))
    print("Baseline mean predictor:", {"mean": baseline, "test_mae": baseline_mae})
    print("Target ranges:", {"train": [float(np.min(y_train)), float(np.max(y_train))], "test": [float(np.min(y_test)), float(np.max(y_test))]})

    out_act = "sigmoid" if str(args.target_col) == "angle_norm" else "linear"
    model = build_tcn_model(
        seq_len=int(args.seq_len),
        n_feat=int(xs.shape[2]),
        out_act=out_act,
        n_blocks=int(args.blocks),
        channels=int(args.channels),
        kernel=int(args.kernel),
        dropout=float(args.dropout),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks: list[tf.keras.callbacks.Callback] = []
    best_path = out_dir / "best.keras"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True))
    if bool(args.early_stop):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(args.patience), restore_best_weights=True))

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, batch_size=int(args.batch_size), verbose=0).reshape(-1)
    corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if y_test.size > 2 else float("nan")

    # Optional per-record calibration evaluation (within the test split)
    calib_metrics = None
    if bool(args.calibrate_per_record):
        if "record" not in full_df.columns:
            raise SystemExit("--calibrate-per-record requires 'record' column.")
        test_records = sorted(set(full_df.loc[test_idx, "record"].astype(str).tolist()))
        all_true: list[float] = []
        all_pred: list[float] = []
        for r in test_records:
            ridx = test_idx[full_df.loc[test_idx, "record"].astype(str).to_numpy() == r]
            if ridx.size < (int(args.calib_windows) + 1):
                continue
            ridx = ridx.astype(np.int64)
            # Use first calib windows for calibration, rest for eval (sequence order is already sorted by t0_s within record)
            order = np.arange(ridx.size)
            cal_local = order[: int(args.calib_windows)]
            ev_local = order[int(args.calib_windows) :]
            cal_idx = ridx[cal_local]
            ev_idx = ridx[ev_local]

            m_cal = tf.keras.models.clone_model(model)
            m_cal.set_weights(model.get_weights())
            # Freeze all but last two dense layers
            for layer in m_cal.layers:
                layer.trainable = False
            for layer in m_cal.layers[::-1]:
                if getattr(layer, "name", "") in {"angle_out", "dense_head"}:
                    layer.trainable = True
                if getattr(layer, "name", "") == "dense_head":
                    break

            m_cal.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.calib_lr)),
                loss=tf.keras.losses.Huber(delta=0.05),
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
            )
            m_cal.fit(xs[cal_idx], ys[cal_idx], epochs=int(args.calib_epochs), batch_size=int(args.batch_size), verbose=0, shuffle=True)
            p = m_cal.predict(xs[ev_idx], batch_size=int(args.batch_size), verbose=0).reshape(-1)
            all_true.extend(ys[ev_idx].tolist())
            all_pred.extend(p.tolist())

        if all_true:
            yt = np.array(all_true, dtype=np.float32)
            yp = np.array(all_pred, dtype=np.float32)
            calib_metrics = {
                "mae": float(np.mean(np.abs(yt - yp))),
                "corr": float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 2 else float("nan"),
                "n": int(yt.size),
                "calib_windows": int(args.calib_windows),
            }

    curves_path = out_dir / "training_curves.png"
    plot_curves(hist, curves_path)
    scatter_path = out_dir / "pred_vs_true.png"
    plot_pred_vs_true(y_test, y_pred, scatter_path)

    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    # Export TFLite for fun / parity
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)

    per_bin = mae_by_bins(y_test.astype(np.float64), y_pred.astype(np.float64), n_bins=20)
    meta = {
        "features_csv": str(features_csv),
        "target_col": str(args.target_col),
        "only_subject": str(args.only_subject) if args.only_subject else None,
        "seq_len": int(args.seq_len),
        "n_features": int(xs.shape[2]),
        "split_by": str(args.split_by),
        "splits": {"val_frac": val_frac, "test_frac": test_frac},
        "baseline": {"mean": baseline, "test_mae": baseline_mae},
        "test_metrics": {"loss": float(test_loss), "mae": float(test_mae), "corr": corr},
        "calibrated_test_metrics": calib_metrics,
        "per_bin_mae": per_bin,
        "training_curves_png": str(curves_path),
        "pred_vs_true_png": str(scatter_path),
        "keras": str(keras_path),
        "tflite": str(tflite_path),
        "tcn": {"blocks": int(args.blocks), "channels": int(args.channels), "kernel": int(args.kernel), "dropout": float(args.dropout)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

