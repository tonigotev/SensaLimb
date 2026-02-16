from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.train import train_conv1d_angle_toro_ossaba as toro


def load_meta(model_dir: Path) -> dict:
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_keras_path(model_dir: Path, meta: dict) -> Path:
    keras_path = meta.get("keras")
    if keras_path:
        p = Path(keras_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return p
    fallback = model_dir / "model.keras"
    if fallback.exists():
        return fallback
    raise SystemExit("Could not locate model.keras. Pass the correct model directory.")


def main() -> None:
    p = argparse.ArgumentParser(description="Export int8 TFLite for Toro Ossaba trained model.")
    p.add_argument("--model-dir", type=str, required=True, help="Folder containing meta.json and model.keras.")
    p.add_argument("--input-root", type=str, default=None, help="Override dataset root (defaults to meta.json input_root).")
    p.add_argument("--rep-samples", type=int, default=500, help="Representative samples for int8 quantization.")
    p.add_argument("--max-windows-per-record", type=int, default=300, help="Limit windows per record for rep data.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-path", type=str, default=None, help="Optional output path for int8 tflite.")
    p.add_argument("--fs", type=float, default=1024.0)
    p.add_argument("--win-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.05)
    p.add_argument("--downsample-hz", type=float, default=200.0)
    p.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    p.add_argument("--envelope-rms-ms", type=float, default=50.0)
    p.add_argument("--window-zscore", action="store_true")
    p.add_argument("--emg-source", choices=["filtered", "raw"], default="filtered")
    p.add_argument("--angle-history-sec", type=float, default=0.0)
    p.add_argument("--target", choices=["angle", "delta"], default="angle")
    p.add_argument("--angle-norm", choices=["minmax", "none"], default="minmax")
    p.add_argument("--angle-min", type=float, default=None)
    p.add_argument("--angle-max", type=float, default=None)
    p.add_argument("--delta-max-deg", type=float, default=None)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    meta = load_meta(model_dir)

    input_root = Path(args.input_root) if args.input_root else Path(meta.get("input_root", "ML/datasets/Toro Ossaba"))
    fs_hz = float(meta.get("fs", args.fs))
    win_sec = float(meta.get("win_sec", args.win_sec))
    step_sec = float(meta.get("step_sec", args.step_sec))
    downsample_hz = float(meta.get("downsample_hz", args.downsample_hz))
    mode = str(meta.get("mode", args.mode))
    envelope_rms_ms = float(meta.get("envelope_rms_ms", args.envelope_rms_ms))
    window_zscore = bool(meta.get("window_zscore", args.window_zscore))
    emg_source = str(meta.get("emg_source", args.emg_source))
    angle_norm = str(meta.get("angle_norm", args.angle_norm))
    angle_min = meta.get("angle_min", args.angle_min)
    angle_max = meta.get("angle_max", args.angle_max)
    angle_history_sec = float(meta.get("angle_history_sec", args.angle_history_sec))
    target = str(meta.get("target", args.target))
    delta_max = meta.get("delta_max_deg", args.delta_max_deg)

    recs = toro.find_toro_files(input_root)
    recs = toro.filter_recs(
        recs,
        only_subject=meta.get("only_subject", None),
        movement=str(meta.get("movement", "all")),
        load=str(meta.get("load", "all")),
    )
    if angle_norm == "minmax" and (angle_min is None or angle_max is None):
        angle_min, angle_max = toro.compute_angle_range(recs)

    items = toro.make_window_index(
        recs,
        fs_hz=fs_hz,
        win_sec=win_sec,
        step_sec=step_sec,
        max_windows_per_record=int(args.max_windows_per_record),
        seed=int(args.seed),
    )

    win = int(round(win_sec * fs_hz))
    ds_factor = 1
    if downsample_hz > 0:
        ds_factor = int(round(fs_hz / float(downsample_hz)))
        ds_factor = max(1, ds_factor)
    input_len = win if mode == "raw" and ds_factor <= 1 else int(max(1, round(win / ds_factor)))
    n_ch = 3 if angle_history_sec > 0 else 2

    def rep_data_gen():
        rep_angle_norm = angle_norm
        rep_angle_min = angle_min
        rep_angle_max = angle_max
        rep_delta_max = delta_max
        if rep_angle_norm != "none":
            if target == "angle" and (rep_angle_min is None or rep_angle_max is None):
                rep_angle_norm = "none"
            if target == "delta" and rep_delta_max is None:
                rep_angle_norm = "none"
        count = 0
        for x, _ in toro.make_dataset_generator(
            items[: max(2000, int(args.rep_samples) * 2)],
            fs_hz=fs_hz,
            win_sec=win_sec,
            mode=mode,
            envelope_rms_ms=envelope_rms_ms,
            downsample_hz=downsample_hz,
            window_zscore=window_zscore,
            emg_source=emg_source,
            angle_norm=rep_angle_norm,
            angle_min=rep_angle_min,
            angle_max=rep_angle_max,
            angle_history_sec=angle_history_sec,
            target=target,
            delta_max=rep_delta_max,
        ):
            x = x[:input_len, :]
            if x.shape[0] < input_len:
                pad = np.zeros((input_len - x.shape[0], x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
            x = x.astype(np.float32, copy=False)
            yield [x.reshape(1, input_len, n_ch)]
            count += 1
            if count >= int(args.rep_samples):
                break

    keras_path = resolve_keras_path(model_dir, meta)
    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()

    out_path = Path(args.out_path) if args.out_path else (model_dir / "model_int8.tflite")
    out_path.write_bytes(tflite_int8)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
