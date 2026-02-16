from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # type: ignore

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.train import train_conv1d_angle_toro_ossaba as toro


def run_tflite(path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    outputs: list[float] = []
    for i in range(inputs.shape[0]):
        x = inputs[i : i + 1]
        if input_details["dtype"] == np.int8:
            scale, zero = input_details["quantization"]
            xq = (x / scale + zero).round().astype(np.int8)
            interpreter.set_tensor(input_details["index"], xq)
        else:
            interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            scale, zero = output_details["quantization"]
            out = (out.astype(np.float32) - zero) * scale
        outputs.append(float(out.reshape(-1)[0]))
    return np.array(outputs, dtype=np.float32)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.25)
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title("Predicted vs True (test)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot pred_vs_true for float and int8 TFLite models.")
    p.add_argument("--model-dir", type=str, required=True, help="Folder containing model.tflite and model_int8.tflite.")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--input-root", type=str, default="ML/datasets/Toro Ossaba")
    p.add_argument("--fs", type=float, default=1000.0)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.05)
    p.add_argument("--downsample-hz", type=float, default=200.0)
    p.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    p.add_argument("--envelope-rms-ms", type=float, default=50.0)
    p.add_argument("--window-zscore", action="store_true")
    p.add_argument("--emg-source", choices=["filtered", "raw"], default="filtered")
    p.add_argument("--angle-history-sec", type=float, default=0.1)
    p.add_argument("--angle-norm", choices=["minmax", "none"], default="minmax")
    p.add_argument("--angle-min", type=float, default=None)
    p.add_argument("--angle-max", type=float, default=None)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    tflite_f32 = model_dir / "model.tflite"
    tflite_int8 = model_dir / "model_int8.tflite"
    if not tflite_f32.exists() or not tflite_int8.exists():
        raise SystemExit("Missing model.tflite or model_int8.tflite in model directory.")

    root = Path(args.input_root)
    recs = toro.find_toro_files(root)
    angle_min = args.angle_min
    angle_max = args.angle_max
    if str(args.angle_norm) == "minmax" and (angle_min is None or angle_max is None):
        angle_min, angle_max = toro.compute_angle_range(recs)
    items = toro.make_window_index(
        recs,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        step_sec=float(args.step_sec),
        max_windows_per_record=500,
        seed=1,
    )

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for x, y in toro.make_dataset_generator(
        items[: int(args.samples)],
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
        target="angle",
        delta_max=None,
    ):
        xs.append(x)
        ys.append(float(y.reshape(-1)[0]))
    if not xs:
        raise SystemExit("No samples generated.")
    xs_arr = np.stack(xs, axis=0).astype(np.float32)
    y_true = np.array(ys, dtype=np.float32)

    y_f32 = run_tflite(tflite_f32, xs_arr)
    y_int8 = run_tflite(tflite_int8, xs_arr)

    label = "angle_norm" if str(args.angle_norm) != "none" else "angle_deg"
    plot_scatter(y_true, y_f32, model_dir / "pred_vs_true_float.png", label)
    plot_scatter(y_true, y_int8, model_dir / "pred_vs_true_int8.png", label)
    print(f"Saved: {model_dir / 'pred_vs_true_float.png'}")
    print(f"Saved: {model_dir / 'pred_vs_true_int8.png'}")


if __name__ == "__main__":
    main()
