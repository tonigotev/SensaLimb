from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_features_csv(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise SystemExit("features.csv must contain a 'label' column.")

    y = df["label"].to_numpy(dtype=np.int64, copy=False)

    # Drop non-feature columns if present
    drop_cols = [c for c in ["subject", "record", "t0_s", "t1_s", "label"] if c in df.columns]
    xdf = df.drop(columns=drop_cols)

    # Ensure numeric
    xdf = xdf.apply(pd.to_numeric, errors="coerce")
    if xdf.isna().any().any():
        # If any NaNs exist, drop those rows (should be rare; better than crashing training)
        mask = ~xdf.isna().any(axis=1)
        xdf = xdf.loc[mask].reset_index(drop=True)
        y = y[mask.to_numpy()]

    x = xdf.to_numpy(dtype=np.float32, copy=False)
    return xdf, y

def make_history(
    full_df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    history: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Concatenate features from previous (history-1) windows within each record, sorted by t0_s.
    This provides temporal context and helps with EMG->pose delay.
    Requires columns: record and t0_s.
    """
    if history <= 1:
        return full_df, x, y
    if "record" not in full_df.columns or "t0_s" not in full_df.columns:
        raise SystemExit(f"--history {history} requires features.csv to include 'record' and 't0_s'")

    out_rows: list[int] = []
    out_x: list[np.ndarray] = []
    out_y: list[int] = []

    groups = full_df["record"].astype(str).to_numpy()
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
            out_y.append(int(y[idx[i]]))

    if not out_rows:
        raise SystemExit(f"--history {history} produced zero samples (not enough windows per record).")

    keep = np.array(out_rows, dtype=np.int64)
    new_df = full_df.loc[keep].reset_index(drop=True)
    new_x = np.stack(out_x, axis=0).astype(np.float32)
    new_y = np.array(out_y, dtype=np.int64)
    return new_df, new_x, new_y


def get_groups(df: pd.DataFrame, split_by: str) -> np.ndarray | None:
    if split_by == "row":
        return None
    if split_by not in df.columns:
        raise SystemExit(f"--split-by {split_by} requested but column '{split_by}' not present in features.csv")
    return df[split_by].astype(str).to_numpy()


def stratified_split(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(y.shape[0])

    train_idx: list[int] = []
    other_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_other = int(round(frac * cls_idx.shape[0]))
        other_idx.extend(cls_idx[:n_other].tolist())
        train_idx.extend(cls_idx[n_other:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    other_idx = np.array(other_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(other_idx)
    return train_idx, other_idx


def plot_curves(history, out_path: Path) -> None:
    hist = history.history
    epochs = np.arange(1, len(next(iter(hist.values()))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in hist:
        axes[0].plot(epochs, hist["loss"], label="train")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0].legend()

    if "acc" in hist:
        axes[1].plot(epochs, hist["acc"], label="train")
    if "val_acc" in hist:
        axes[1].plot(epochs, hist["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def group_split_indices(groups: np.ndarray, val_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by group (record/subject) to avoid leakage.
    Not stratified by label; we print per-split class counts to help diagnose imbalance.
    """
    rng = np.random.default_rng(seed)
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
        mask = np.isin(groups, np.array(gs, dtype=object))
        return int(np.sum(mask))

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
    return train_idx, val_idx, test_idx


def stratified_group_split_indices(
    groups: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group split (record/subject) while trying to preserve label distribution.

    Heuristic: compute per-group class histograms, then greedily assign groups to test/val/train
    to match the global class proportions.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)

    # Per-group class counts
    group_counts: dict[str, np.ndarray] = {}
    for g in uniq:
        mask = groups == g
        group_counts[str(g)] = np.bincount(y[mask].astype(int), minlength=n_classes)

    total = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    total_rows = int(np.sum(total))
    target_test = total * float(test_frac)
    target_val = total * float(val_frac)

    # Sort groups by size descending for better greedy packing
    group_list = list(group_counts.keys())
    group_list.sort(key=lambda k: int(np.sum(group_counts[k])), reverse=True)
    rng.shuffle(group_list)  # add randomness among same-sized groups

    test_groups: list[str] = []
    val_groups: list[str] = []
    train_groups: list[str] = []

    sum_test = np.zeros(n_classes, dtype=np.float64)
    sum_val = np.zeros(n_classes, dtype=np.float64)

    def score_add(current: np.ndarray, target: np.ndarray, add: np.ndarray) -> float:
        # L1 distance after adding
        return float(np.sum(np.abs((current + add) - target)))

    for g in group_list:
        cnt = group_counts[g].astype(np.float64)

        # Prefer assigning to the split where it improves target matching most,
        # while respecting rough size goals. If both test/val are "full enough",
        # assign remaining groups to train.
        test_ok = np.sum(sum_test) < (test_frac * total_rows * 1.05)
        val_ok = np.sum(sum_val) < (val_frac * total_rows * 1.05)

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
        else:
            train_groups.append(g)

    test_mask = np.isin(groups, np.array(test_groups, dtype=object))
    val_mask = np.isin(groups, np.array(val_groups, dtype=object)) & ~test_mask
    train_mask = ~(test_mask | val_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def pick_calibration_records_for_subject(
    df: pd.DataFrame,
    subject: str,
    indices: np.ndarray,
    *,
    n_records: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From indices belonging to a single subject, pick n_records unique record names for calibration,
    return (calib_idx, eval_idx) (both are global indices).
    """
    sub_mask = (df["subject"].astype(str).to_numpy() == subject)
    sub_idx = indices[sub_mask[indices]]
    if sub_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    records = df.loc[sub_idx, "record"].astype(str).unique().tolist()
    if not records:
        return np.array([], dtype=np.int64), sub_idx.astype(np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(records)
    picked = set(records[: max(1, int(n_records))])

    calib_mask = df.loc[sub_idx, "record"].astype(str).isin(picked).to_numpy()
    calib_idx = sub_idx[calib_mask]
    eval_idx = sub_idx[~calib_mask]
    return calib_idx.astype(np.int64), eval_idx.astype(np.int64)


def pick_calibration_records_covering_classes(
    df: pd.DataFrame,
    subject: str,
    indices: np.ndarray,
    *,
    n_records_max: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick up to n_records_max records for calibration for this subject trying to cover all classes.
    Falls back to 1 record if coverage isn't possible.
    """
    sub_mask = (df["subject"].astype(str).to_numpy() == subject)
    sub_idx = indices[sub_mask[indices]]
    if sub_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    recs = df.loc[sub_idx, "record"].astype(str).unique().tolist()
    if not recs:
        return np.array([], dtype=np.int64), sub_idx.astype(np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(recs)

    # Precompute per-record class presence
    rec_classes: dict[str, set[int]] = {}
    for r in recs:
        ridx = sub_idx[df.loc[sub_idx, "record"].astype(str).to_numpy() == r]
        labels = df.loc[ridx, "label"].astype(int).to_numpy()
        present = set(int(x) for x in np.unique(labels) if 0 <= int(x) < n_classes)
        rec_classes[r] = present

    chosen: list[str] = []
    covered: set[int] = set()
    for _ in range(max(1, int(n_records_max))):
        best_r = None
        best_gain = -1
        for r in recs:
            if r in chosen:
                continue
            gain = len(rec_classes[r] - covered)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_r is None:
            break
        chosen.append(best_r)
        covered |= rec_classes[best_r]
        if len(covered) == n_classes:
            break

    calib_mask = df.loc[sub_idx, "record"].astype(str).isin(set(chosen)).to_numpy()
    calib_idx = sub_idx[calib_mask]
    eval_idx = sub_idx[~calib_mask]
    return calib_idx.astype(np.int64), eval_idx.astype(np.int64)


def clone_and_freeze_for_calibration(model, trainable: str):
    import tensorflow as tf  # type: ignore

    m = tf.keras.models.clone_model(model)
    m.set_weights(model.get_weights())

    if trainable == "all":
        for layer in m.layers:
            layer.trainable = True
    else:
        for layer in m.layers:
            layer.trainable = False
        # last layer is class_probs
        for layer in m.layers[::-1]:
            if getattr(layer, "name", "") == "class_probs":
                layer.trainable = True
                break
        else:  # pragma: no cover
            # fallback: last layer
            m.layers[-1].trainable = True

    return m


def adapt_normalization_layer(m, x_cal: np.ndarray) -> None:
    """
    Re-adapt the first Normalization layer on calibration data to reduce session shift.
    """
    import tensorflow as tf  # type: ignore

    for layer in m.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.adapt(x_cal)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3-class MLP on Mendeley sEMG window features and export TFLite.")
    parser.add_argument(
        "--features-csv",
        type=str,
        default="ML/datasets/Mendeley/sEMG_recordings/normalized/labeled/features.csv",
        help="Path to features.csv produced by label_mendeley_semg.py",
    )
    parser.add_argument("--out-dir", type=str, default="ML/models/mlp_3class", help="Output directory")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1, help="Hold out fraction for test set (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden",
        type=str,
        default="256,128,64",
        help='Comma-separated hidden layer sizes, e.g. "256,128,64" (default: 256,128,64).',
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate between dense layers (default: 0.2).")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 weight decay for Dense kernels (default: 0).")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping on val_loss.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs). Default: 15.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Apply class weights during training to reduce bias (default: none).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the loss (0 disables). Typical: 0.05.",
    )
    parser.add_argument(
        "--split-by",
        choices=["row", "record", "subject"],
        default="row",
        help="How to split data. 'row' is random stratified by class; 'record'/'subject' avoid leakage (default: row).",
    )
    parser.add_argument("--history", type=int, default=1, help="Number of past windows to concatenate (default: 1).")
    parser.add_argument(
        "--calibrate-per-subject",
        action="store_true",
        help=(
            "When --split-by subject, evaluate with per-subject calibration: "
            "hold out N records per test subject as calibration, fine-tune briefly, then test on remaining records."
        ),
    )
    parser.add_argument("--calib-records", type=int, default=1, help="Calibration records per test subject (default: 1).")
    parser.add_argument("--calib-epochs", type=int, default=10, help="Fine-tune epochs on calibration data (default: 10).")
    parser.add_argument("--calib-lr", type=float, default=1e-4, help="Fine-tune learning rate (default: 1e-4).")
    parser.add_argument(
        "--calib-trainable",
        choices=["last", "all"],
        default="last",
        help="Which layers are trainable during calibration (default: last).",
    )
    parser.add_argument("--quantize-int8", action="store_true", help="Export int8 quantized TFLite using representative data.")
    args = parser.parse_args()

    # Import tensorflow lazily so this script can exist even if TF isn't installed.
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow is required for training/export.\n"
            "Install it in a separate environment (recommended Python 3.11/3.12).\n"
            f"Original import error: {exc}"
        )

    features_csv = Path(args.features_csv)
    if not features_csv.is_file():
        raise SystemExit(f"features.csv not found: {features_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(features_csv)
    xdf, y = load_features_csv(features_csv)
    x = xdf.to_numpy(dtype=np.float32, copy=False)

    # Optional history stacking (sequence context)
    full_df, x, y = make_history(full_df, x, y, history=int(args.history))

    classes = sorted(set(int(v) for v in np.unique(y)))
    if classes != list(range(len(classes))):
        raise SystemExit(f"Expected labels 0..K-1. Got labels: {classes}")
    n_classes = len(classes)
    if n_classes != 3:
        raise SystemExit(f"Expected 3 classes, got {n_classes} (labels: {classes})")

    # Split
    test_frac = float(args.test_frac)
    val_frac = float(args.val_frac)
    if not (0.0 < test_frac < 0.5):
        raise SystemExit("--test-frac must be in (0, 0.5)")
    if not (0.0 < val_frac < 0.5):
        raise SystemExit("--val-frac must be in (0, 0.5)")
    if (test_frac + val_frac) >= 0.8:
        raise SystemExit("val+test fractions too large; keep them reasonable (e.g. 0.2 + 0.1).")

    groups = get_groups(full_df, str(args.split_by))
    if groups is None:
        # Stratified split: train / temp, then temp -> val / test
        train_idx, temp_idx = stratified_split(y, test_frac + val_frac, int(args.seed))
        y_temp = y[temp_idx]
        # Split temp into val/test, preserving relative proportion
        rel_test = test_frac / (test_frac + val_frac)
        val_idx, test_idx = stratified_split(y_temp, rel_test, int(args.seed) + 1)
        # val_idx/test_idx are indices into temp_idx
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
    else:
        train_idx, val_idx, test_idx = stratified_group_split_indices(
            groups,
            y,
            n_classes,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=int(args.seed),
        )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    def split_stats(name: str, yy: np.ndarray) -> dict[str, object]:
        counts = np.bincount(yy.astype(int), minlength=n_classes).tolist() if yy.size else [0] * n_classes
        return {"n": int(yy.size), "class_counts": counts}

    stats = {
        "train": split_stats("train", y_train),
        "val": split_stats("val", y_val),
        "test": split_stats("test", y_test),
    }
    print("Split stats:", json.dumps(stats, indent=2))

    if y_val.size == 0 or y_test.size == 0:
        raise SystemExit(
            "Validation or test split ended up empty. "
            "Try increasing dataset size, reducing --val-frac/--test-frac, or using --split-by row."
        )

    # Normalization layer (saved in the model; important for deployment parity)
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(x_train)

    hidden = [int(s.strip()) for s in str(args.hidden).split(",") if s.strip()]
    if not hidden:
        raise SystemExit("--hidden must contain at least one layer size, e.g. 256,128,64")
    if any(h <= 0 for h in hidden):
        raise SystemExit("--hidden values must be positive integers")

    kernel_reg = None
    if float(args.l2) > 0:
        kernel_reg = tf.keras.regularizers.L2(float(args.l2))

    layers: list[tf.keras.layers.Layer] = [
        tf.keras.Input(shape=(x.shape[1],), name="features"),
        norm,
    ]
    for i, h in enumerate(hidden):
        layers.append(
            tf.keras.layers.Dense(
                h,
                activation="relu",
                kernel_regularizer=kernel_reg,
                name=f"dense_{i}_{h}",
            )
        )
        if float(args.dropout) > 0:
            layers.append(tf.keras.layers.Dropout(float(args.dropout), name=f"dropout_{i}"))
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probs"))

    model = tf.keras.Sequential(layers)

    label_smoothing = float(args.label_smoothing)
    if label_smoothing < 0 or label_smoothing >= 1:
        raise SystemExit("--label-smoothing must be in [0, 1). Typical: 0.05")

    use_onehot = label_smoothing > 0
    if use_onehot:
        y_train_fit = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_val_fit = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        y_test_fit = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    else:
        y_train_fit = y_train
        y_val_fit = y_val
        y_test_fit = y_test
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=loss_fn,
        metrics=metrics,
    )

    callbacks: list[tf.keras.callbacks.Callback] = []
    best_path = out_dir / "best.keras"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True))
    if args.early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(args.patience),
                restore_best_weights=True,
            )
        )

    class_weight = None
    if args.class_weight == "balanced":
        # inverse-frequency weights from training labels only
        counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(np.float64)
        weights = (np.sum(counts) / (counts + 1e-9)) / n_classes
        class_weight = {i: float(weights[i]) for i in range(n_classes)}
        print("Class weights:", class_weight)

    history = model.fit(
        x_train,
        y_train_fit,
        validation_data=(x_val, y_val_fit),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=2,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # Evaluate on held-out test set
    test_loss, test_acc = model.evaluate(x_test, y_test_fit, verbose=0)

    # Confusion matrix + per-class accuracy on test
    probs = model.predict(x_test, batch_size=int(args.batch_size), verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(np.int64)
    cm = confusion_matrix(y_test, y_pred, n_classes=n_classes)
    cm_path = out_dir / "confusion_matrix_test.png"
    class_names = ["straight", "mid_90", "full_flex"]
    plot_confusion_matrix(cm, cm_path, class_names)
    per_class_acc = {}
    for c in range(n_classes):
        denom = int(np.sum(cm[c, :]))
        per_class_acc[str(c)] = float(cm[c, c] / denom) if denom > 0 else float("nan")

    # Optional: per-subject calibration evaluation (only meaningful when splitting by subject)
    calib_metrics = None
    calib_cm = None
    calib_cm_path = None
    if bool(args.calibrate_per_subject):
        if str(args.split_by) != "subject":
            raise SystemExit("--calibrate-per-subject requires --split-by subject")
        if "subject" not in full_df.columns or "record" not in full_df.columns:
            raise SystemExit("features.csv must include 'subject' and 'record' columns for calibration.")

        test_subjects = sorted(full_df.loc[test_idx, "subject"].astype(str).unique().tolist())
        all_preds: list[int] = []
        all_true: list[int] = []

        for subj in test_subjects:
            calib_idx, eval_idx = pick_calibration_records_covering_classes(
                full_df,
                subj,
                test_idx,
                n_records_max=int(args.calib_records),
                n_classes=n_classes,
                seed=int(args.seed) + (abs(hash(subj)) % 10_000),
            )
            if calib_idx.size == 0 or eval_idx.size == 0:
                continue

            x_cal = x[calib_idx]
            y_cal = y[calib_idx]
            x_ev = x[eval_idx]
            y_ev = y[eval_idx]

            # Build calibrated copy
            m_cal = clone_and_freeze_for_calibration(model, str(args.calib_trainable))
            # Update normalization stats for this subject/session
            adapt_normalization_layer(m_cal, x_cal)

            # During calibration, avoid label smoothing (it can encourage collapse on small calibration sets)
            if use_onehot:
                loss_fn_cal = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
            else:
                loss_fn_cal = tf.keras.losses.SparseCategoricalCrossentropy()

            m_cal.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.calib_lr)),
                loss=loss_fn_cal,
                metrics=metrics,
            )
            if use_onehot:
                y_cal_fit = tf.keras.utils.to_categorical(y_cal, num_classes=n_classes)
                y_ev_fit = tf.keras.utils.to_categorical(y_ev, num_classes=n_classes)
            else:
                y_cal_fit = y_cal
                y_ev_fit = y_ev

            m_cal.fit(
                x_cal,
                y_cal_fit,
                epochs=int(args.calib_epochs),
                batch_size=int(args.batch_size),
                verbose=0,
                shuffle=True,
            )

            # Evaluate/predict on remaining records for this subject
            p = m_cal.predict(x_ev, batch_size=int(args.batch_size), verbose=0)
            yp = np.argmax(p, axis=1).astype(np.int64)
            all_preds.extend(yp.tolist())
            all_true.extend(y_ev.astype(int).tolist())

        if all_true:
            y_true_cal = np.array(all_true, dtype=np.int64)
            y_pred_cal = np.array(all_preds, dtype=np.int64)
            calib_cm = confusion_matrix(y_true_cal, y_pred_cal, n_classes=n_classes)
            calib_cm_path = out_dir / "confusion_matrix_test_calibrated.png"
            plot_confusion_matrix(calib_cm, calib_cm_path, class_names)
            calib_acc = float(np.mean(y_true_cal == y_pred_cal))
            calib_metrics = {"acc": calib_acc, "n": int(y_true_cal.size)}

    # Plot curves
    curves_path = out_dir / "training_curves.png"
    plot_curves(history, curves_path)

    # Save Keras model
    keras_path = out_dir / "model.keras"
    model.save(keras_path)

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quantize_int8:
        # Representative dataset for int8 calibration
        def rep_data():
            # Use a small random subset of training data
            n = min(2000, x_train.shape[0])
            for i in range(0, n, 1):
                yield [x_train[i : i + 1]]

        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = out_dir / ("model_int8.tflite" if args.quantize_int8 else "model.tflite")
    tflite_path.write_bytes(tflite_model)

    # Save metadata needed for deployment
    meta = {
        "features_csv": str(features_csv),
        "n_features": int(x.shape[1]),
        "feature_columns": list(xdf.columns),
        "history": int(args.history),
        "n_classes": int(n_classes),
        "label_mapping": {0: "straight", 1: "mid_90", 2: "full_flex"},
        "history_last": {k: float(v[-1]) for k, v in history.history.items()},
        "splits": {"val_frac": val_frac, "test_frac": test_frac},
        "test_metrics": {"loss": float(test_loss), "acc": float(test_acc)},
        "test_confusion_matrix": cm.tolist(),
        "test_per_class_acc": per_class_acc,
        "confusion_matrix_png": str(cm_path),
        "training_curves_png": str(curves_path),
        "calibrated_test_metrics": calib_metrics,
        "calibrated_test_confusion_matrix": calib_cm.tolist() if calib_cm is not None else None,
        "calibrated_confusion_matrix_png": str(calib_cm_path) if calib_cm_path is not None else None,
        "tflite": str(tflite_path),
        "keras": str(keras_path),
        "quantized_int8": bool(args.quantize_int8),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved Keras model: {keras_path}")
    print(f"Saved TFLite model: {tflite_path}")
    print(f"Saved metadata: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

