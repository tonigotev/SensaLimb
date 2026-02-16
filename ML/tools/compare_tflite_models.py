from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    p = argparse.ArgumentParser(description="Compare float vs int8 TFLite outputs.")
    p.add_argument("--model-dir", type=str, required=True, help="Folder containing model.tflite and model_int8.tflite.")
    p.add_argument("--samples", type=int, default=50)
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
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    tflite_f32 = model_dir / "model.tflite"
    tflite_int8 = model_dir / "model_int8.tflite"
    if not tflite_f32.exists() or not tflite_int8.exists():
        raise SystemExit("Missing model.tflite or model_int8.tflite in model directory.")

    root = Path(args.input_root)
    recs = toro.find_toro_files(root)
    items = toro.make_window_index(
        recs,
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        step_sec=float(args.step_sec),
        max_windows_per_record=50,
        seed=1,
    )

    xs: list[np.ndarray] = []
    for x, _ in toro.make_dataset_generator(
        items[: int(args.samples)],
        fs_hz=float(args.fs),
        win_sec=float(args.win_sec),
        mode=str(args.mode),
        envelope_rms_ms=float(args.envelope_rms_ms),
        downsample_hz=float(args.downsample_hz),
        window_zscore=bool(args.window_zscore),
        emg_source=str(args.emg_source),
        angle_norm="none",
        angle_min=None,
        angle_max=None,
        angle_history_sec=float(args.angle_history_sec),
        target="angle",
        delta_max=None,
    ):
        xs.append(x)
    if not xs:
        raise SystemExit("No samples generated.")
    xs = np.stack(xs, axis=0).astype(np.float32)

    y_f32 = run_tflite(tflite_f32, xs)
    y_int8 = run_tflite(tflite_int8, xs)
    diff = np.abs(y_f32 - y_int8)
    print("mean abs diff:", float(diff.mean()))
    print("max abs diff:", float(diff.max()))


if __name__ == "__main__":
    main()
