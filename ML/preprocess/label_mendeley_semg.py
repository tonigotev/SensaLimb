from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Hysteresis:
    """
    Two-state hysteresis for angle -> label.

    label 0: extended (triceps position)
    label 1: flexed   (biceps position)

    - Enter/keep extended when angle <= low_enter, stay extended until angle >= low_exit
    - Enter/keep flexed   when angle >= high_enter, stay flexed   until angle <= high_exit
    """

    low_enter: float = 20.0
    low_exit: float = 30.0
    high_enter: float = 120.0
    high_exit: float = 110.0

    def validate(self) -> None:
        if not (self.low_enter <= self.low_exit):
            raise ValueError("Require low_enter <= low_exit")
        if not (self.high_exit <= self.high_enter):
            raise ValueError("Require high_exit <= high_enter")
        if not (self.low_exit < self.high_exit):
            # otherwise the bands overlap and state can flip unpredictably
            raise ValueError("Require low_exit < high_exit (non-overlapping hysteresis bands)")


@dataclass(frozen=True)
class ThreeStateHysteresis:
    """
    Three-state hysteresis for angle -> label.

    label 0: straight/extended
    label 1: mid flex (e.g. ~90 degrees)
    label 2: full flex

    Uses two hysteresis boundaries:
      - between 0 and 1: 0->1 when angle >= t01_up, 1->0 when angle <= t10_down
      - between 1 and 2: 1->2 when angle >= t12_up, 2->1 when angle <= t21_down
    """

    t01_up: float = 50.0
    t10_down: float = 30.0
    t12_up: float = 110.0
    t21_down: float = 100.0

    def validate(self) -> None:
        if not (self.t10_down < self.t01_up):
            raise ValueError("Require t10_down < t01_up (hysteresis for boundary 0<->1)")
        if not (self.t21_down < self.t12_up):
            raise ValueError("Require t21_down < t12_up (hysteresis for boundary 1<->2)")
        if not (self.t01_up < self.t21_down):
            raise ValueError("Require t01_up < t21_down (non-overlapping boundaries)")


def find_recording_files(root: Path) -> list[Path]:
    return sorted(set(root.glob("Subject_*/*.csv")) | set(root.glob("subject_*/*.csv")))


def infer_subject_name(csv_path: Path) -> str:
    return csv_path.parent.name


def load_three_column_csv(file_path: Path, *, chunksize: int | None = None):
    """
    Streaming CSV reader. Returns an iterator of DataFrames if chunksize is set,
    otherwise returns a single DataFrame.
    """
    return pd.read_csv(
        file_path,
        sep=",",
        engine="c",
        header=None,
        usecols=[0, 1, 2],
        comment="#",
        on_bad_lines="skip",
        chunksize=chunksize,
    )


def hysteresis_labels(angle_deg: np.ndarray, h: Hysteresis, *, initial: int | None = None) -> np.ndarray:
    """
    Convert angle series to 0/1 labels using hysteresis and a state machine.
    All samples will be labeled.
    """
    h.validate()
    a = angle_deg.astype(float, copy=False)
    n = a.shape[0]
    out = np.empty(n, dtype=np.int8)

    # Initialize state
    if initial in (0, 1):
        state = int(initial)
    else:
        # choose based on first sample proximity / thresholds
        if a[0] <= h.low_exit:
            state = 0
        elif a[0] >= h.high_exit:
            state = 1
        else:
            # ambiguous: pick nearest boundary
            state = 0 if abs(a[0] - h.low_exit) <= abs(a[0] - h.high_exit) else 1

    for i in range(n):
        x = a[i]
        if state == 0:
            # extended: only switch to flexed after clearly entering high band
            if x >= h.high_enter:
                state = 1
        else:
            # flexed: only switch to extended after clearly entering low band
            if x <= h.low_enter:
                state = 0
        # If we are in-between, we keep the current state (that's the hysteresis).
        out[i] = state

    return out


def hysteresis_labels_3state(
    angle_deg: np.ndarray, h: ThreeStateHysteresis, *, initial: int | None = None
) -> np.ndarray:
    """
    Convert angle series to 0/1/2 labels using two hysteresis boundaries.
    All samples will be labeled.
    """
    h.validate()
    a = angle_deg.astype(float, copy=False)
    n = a.shape[0]
    out = np.empty(n, dtype=np.int8)

    # Initialize state
    if initial in (0, 1, 2):
        state = int(initial)
    else:
        b01_mid = (h.t10_down + h.t01_up) / 2.0
        b12_mid = (h.t21_down + h.t12_up) / 2.0
        if a[0] <= b01_mid:
            state = 0
        elif a[0] >= b12_mid:
            state = 2
        else:
            state = 1

    for i in range(n):
        x = a[i]
        if state == 0:
            if x >= h.t01_up:
                state = 1
        elif state == 1:
            if x <= h.t10_down:
                state = 0
            elif x >= h.t12_up:
                state = 2
        else:
            if x <= h.t21_down:
                state = 1
        out[i] = state

    return out


def bin_labels_3state(angle_deg: np.ndarray, *, straight_max: float, mid_max: float) -> np.ndarray:
    """
    Simple 3-bin labeling:
      0 if angle <= straight_max
      1 if straight_max < angle <= mid_max
      2 if angle > mid_max
    """
    if not (straight_max < mid_max):
        raise ValueError("Require straight_max < mid_max")
    a = angle_deg.astype(float, copy=False)
    out = np.empty(a.shape[0], dtype=np.int8)
    out[a <= straight_max] = 0
    out[(a > straight_max) & (a <= mid_max)] = 1
    out[a > mid_max] = 2
    return out


def window_features(x: np.ndarray) -> dict[str, float]:
    """
    Simple time-domain features for EMG window x (1D).
    """
    x = x.astype(float, copy=False)
    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    var = float(np.var(x))
    wl = float(np.sum(np.abs(np.diff(x))))
    return {"mav": mav, "rms": rms, "var": var, "wl": wl}


def pair_features(b: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """
    Cross-channel features (biceps vs triceps) for a single window.
    These often help separate poses because relative activation matters more than absolute.
    """
    eps = 1e-9
    fb = window_features(b)
    ft = window_features(t)

    # Ratios and differences of key amplitudes
    out = {
        "rms_ratio": float(fb["rms"] / (ft["rms"] + eps)),
        "mav_ratio": float(fb["mav"] / (ft["mav"] + eps)),
        "wl_ratio": float(fb["wl"] / (ft["wl"] + eps)),
        "rms_diff": float(fb["rms"] - ft["rms"]),
        "mav_diff": float(fb["mav"] - ft["mav"]),
        "wl_diff": float(fb["wl"] - ft["wl"]),
    }

    # Correlation (bounded [-1, 1]) can capture co-contraction patterns
    bb = b.astype(float, copy=False)
    tt = t.astype(float, copy=False)
    bb = bb - float(np.mean(bb))
    tt = tt - float(np.mean(tt))
    denom = float(np.sqrt(np.sum(bb * bb) * np.sum(tt * tt)) + eps)
    out["corr"] = float(np.sum(bb * tt) / denom)

    return out

def fft_bin_features(
    x: np.ndarray,
    *,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    bin_hz: float,
    log_power: bool = True,
) -> dict[str, float]:
    """
    FFT power features aggregated into frequency bins.

    For a window of N samples, raw FFT bin spacing is df = fs/N.
    To reduce dimensionality and stabilize features, we average power inside bins of width bin_hz.

    Returns features named like: fft_20_30 (meaning [20,30) Hz).
    """
    x = x.astype(float, copy=False)
    n = x.shape[0]
    if n < 8:
        return {}

    # Remove DC bias in the window (helps prevent low-freq dominance).
    x = x - float(np.mean(x))

    # Hann window reduces spectral leakage (important for short windows).
    w = np.hanning(n)
    xw = x * w

    # One-sided FFT power spectrum
    spec = np.fft.rfft(xw)
    power = (spec.real * spec.real + spec.imag * spec.imag) / max(n, 1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    # Limit to desired frequency range
    fmin_hz = float(fmin_hz)
    fmax_hz = float(fmax_hz)
    bin_hz = float(bin_hz)
    if bin_hz <= 0:
        raise ValueError("bin_hz must be > 0")
    if fmax_hz <= fmin_hz:
        raise ValueError("Require fmax_hz > fmin_hz")

    feats: dict[str, float] = {}
    start = fmin_hz
    eps = 1e-20
    while start < fmax_hz:
        end = min(start + bin_hz, fmax_hz)
        mask = (freqs >= start) & (freqs < end)
        if not np.any(mask):
            val = 0.0
        else:
            val = float(np.mean(power[mask]))
        if log_power:
            val = float(np.log10(val + eps))
        key = f"fft_{int(round(start))}_{int(round(end))}"
        feats[key] = val
        start = end

    return feats


def majority_label(labels: np.ndarray, n_classes: int) -> int:
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    return int(np.argmax(counts))


def window_velocity_dps(
    angle_window: np.ndarray,
    *,
    fs_hz: float,
    edge_ms: float = 20.0,
) -> float:
    """
    Estimate angular velocity (deg/s) for a window using start/end averages to reduce noise.
    """
    a = angle_window.astype(float, copy=False)
    n = int(a.shape[0])
    if n < 2:
        return 0.0
    k = int(round((edge_ms / 1000.0) * fs_hz))
    k = max(1, min(k, n // 2))
    a0 = float(np.mean(a[:k]))
    a1 = float(np.mean(a[-k:]))
    dt = float(n / fs_hz)
    if dt <= 0:
        return 0.0
    return (a1 - a0) / dt


def window_velocity_slope_dps(angle_window: np.ndarray, *, fs_hz: float) -> float:
    """
    Estimate angular velocity (deg/s) by fitting a line to angle vs time within the window.
    This is usually smoother than start/end differencing when angles are quantized.
    """
    a = angle_window.astype(float, copy=False)
    n = int(a.shape[0])
    if n < 3:
        return 0.0

    # time axis in seconds, centered to reduce numerical issues
    t = (np.arange(n, dtype=np.float64) / float(fs_hz))
    t = t - float(np.mean(t))
    aa = a.astype(np.float64)

    # slope = cov(t,a) / var(t)
    denom = float(np.sum(t * t))
    if denom <= 0:
        return 0.0
    slope = float(np.sum(t * (aa - float(np.mean(aa)))) / denom)
    return slope

def make_windows(n: int, win: int, step: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i + win <= n:
        out.append((i, i + win))
        i += step
    return out


def label_one_file(
    csv_path: Path,
    out_dir: Path,
    *,
    n_states: int,
    h2: Hysteresis | None = None,
    h3: ThreeStateHysteresis | None = None,
    label_mode: str = "hysteresis",
    bin_straight_max: float = 40.0,
    bin_mid_max: float = 80.0,
    chunksize: int = 200_000,
    export_labeled_csv: bool = True,
    export_features_csv: bool = False,
    fs_hz: float = 2000.0,
    win_sec: float = 0.2,
    step_sec: float = 0.05,
    export_fft: bool = False,
    fft_fmin_hz: float = 20.0,
    fft_fmax_hz: float = 450.0,
    fft_bin_hz: float = 10.0,
    fft_log_power: bool = True,
    min_label_frac: float = 0.0,
    target_mode: str = "class",
    vel_max_dps: float = 300.0,
    vel_edge_ms: float = 20.0,
    vel_method: str = "edge",
    min_abs_vel_norm: float = 0.0,
    angle_max_deg: float = 150.0,
    angle_stat: str = "mean",
) -> None:
    """
    Writes:
      - <out_dir>/<subject>/<stem>_labeled.csv with 4 columns: biceps,triceps,angle,label
      - optionally appends per-window features to <out_dir>/features.csv
    """
    subj = infer_subject_name(csv_path).lower()
    per_dir = out_dir / subj
    per_dir.mkdir(parents=True, exist_ok=True)

    out_labeled = per_dir / f"{csv_path.stem}_labeled.csv"

    win = int(round(win_sec * fs_hz))
    step = int(round(step_sec * fs_hz))
    if export_features_csv and (win < 2 or step < 1):
        raise ValueError("Invalid win/step for features.")
    if not (0.0 <= min_label_frac <= 1.0):
        raise ValueError("min_label_frac must be in [0, 1].")
    if float(vel_max_dps) <= 0:
        raise ValueError("vel_max_dps must be > 0")
    if float(vel_edge_ms) <= 0:
        raise ValueError("vel_edge_ms must be > 0")
    if vel_method not in {"edge", "slope"}:
        raise ValueError("vel_method must be one of: edge, slope")
    if not (0.0 <= float(min_abs_vel_norm) < 1.0):
        raise ValueError("min_abs_vel_norm must be in [0, 1)")
    if float(angle_max_deg) <= 0:
        raise ValueError("angle_max_deg must be > 0")
    if angle_stat not in {"mean", "end"}:
        raise ValueError("angle_stat must be one of: mean, end")

    feature_rows: list[dict[str, object]] = []

    # Stream in chunks; for labeled CSV we can write per chunk.
    reader = load_three_column_csv(csv_path, chunksize=chunksize)
    first_write = True

    # NOTE: For window features, we need continuity across chunks.
    carry: np.ndarray | None = None
    carry_labels: np.ndarray | None = None
    carry_offset = 0  # sample index offset for carry

    sample_index = 0
    prev_state: int | None = None

    for df in reader:
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            continue
        arr = df.to_numpy(dtype=float, copy=False)  # (N,3)

        # Labels for this chunk, using previous state to keep continuity
        angle = arr[:, 2]
        if n_states == 2:
            if label_mode != "hysteresis":
                raise SystemExit("label_mode 'bins' is only implemented for 3-state right now.")
            if h2 is None:
                raise ValueError("h2 must be provided for 2-state labeling")
            labels = hysteresis_labels(angle, h2, initial=prev_state)
            prev_state = int(labels[-1])
        else:
            if label_mode == "bins":
                labels = bin_labels_3state(angle, straight_max=bin_straight_max, mid_max=bin_mid_max)
                prev_state = None
            else:
                if h3 is None:
                    raise ValueError("h3 must be provided for 3-state labeling")
                labels = hysteresis_labels_3state(angle, h3, initial=prev_state)
                prev_state = int(labels[-1])

        if export_labeled_csv:
            out_chunk = np.column_stack([arr, labels])
            pd.DataFrame(out_chunk).to_csv(out_labeled, index=False, header=False, mode="w" if first_write else "a")
            first_write = False

        if export_features_csv:
            # Build a continuous buffer: carry + current
            if carry is None:
                buf = arr
                buf_labels = labels
                buf_start_idx = sample_index
            else:
                buf = np.vstack([carry, arr])
                buf_labels = np.concatenate([carry_labels, labels])  # type: ignore[arg-type]
                buf_start_idx = carry_offset

            nbuf = buf.shape[0]
            windows = make_windows(nbuf, win, step)

            # Only emit windows that fully lie before the last "incomplete tail" we'll carry forward.
            # We keep the last (win-1) samples to not miss windows across boundaries.
            tail_keep = max(win - 1, 0)
            max_emit_end = max(nbuf - tail_keep, 0)

            for s, e in windows:
                if e > max_emit_end:
                    break
                b = buf[s:e, 0]
                t = buf[s:e, 1]
                a = buf[s:e, 2]
                y = buf_labels[s:e]

                # Drop "transition" windows: require a dominant label inside the window.
                if min_label_frac > 0:
                    counts = np.bincount(y.astype(int), minlength=n_states)
                    dom = float(np.max(counts) / max(len(y), 1))
                    if dom < min_label_frac:
                        continue

                row: dict[str, object] = {
                    "subject": infer_subject_name(csv_path),
                    "record": csv_path.stem,
                    "t0_s": (buf_start_idx + s) / fs_hz,
                    "t1_s": (buf_start_idx + e) / fs_hz,
                }
                if target_mode == "class":
                    row["label"] = majority_label(y, n_states)
                elif target_mode == "velocity":
                    if vel_method == "slope":
                        v_dps = window_velocity_slope_dps(a, fs_hz=fs_hz)
                    else:
                        v_dps = window_velocity_dps(a, fs_hz=fs_hz, edge_ms=vel_edge_ms)
                    v_norm = float(np.clip(v_dps / float(vel_max_dps), -1.0, 1.0))
                    if abs(v_norm) < float(min_abs_vel_norm):
                        continue
                    row["vel_dps"] = v_dps
                    row["vel_norm"] = v_norm
                elif target_mode == "angle":
                    # Predict position (angle) instead of velocity.
                    # Angle is in degrees (we keep it un-normalized in the normalized dataset by design).
                    if angle_stat == "end":
                        a_deg = float(a[-1])
                    else:
                        a_deg = float(np.mean(a))
                    a_norm = float(np.clip(a_deg / float(angle_max_deg), 0.0, 1.0))
                    row["angle_deg"] = a_deg
                    row["angle_norm"] = a_norm
                else:
                    raise ValueError(f"Unknown target_mode: {target_mode}")

                row.update({f"biceps_{k}": v for k, v in window_features(b).items()})
                row.update({f"triceps_{k}": v for k, v in window_features(t).items()})
                row.update({f"pair_{k}": v for k, v in pair_features(b, t).items()})
                if export_fft:
                    row.update(
                        {
                            f"biceps_{k}": v
                            for k, v in fft_bin_features(
                                b,
                                fs_hz=fs_hz,
                                fmin_hz=fft_fmin_hz,
                                fmax_hz=fft_fmax_hz,
                                bin_hz=fft_bin_hz,
                                log_power=fft_log_power,
                            ).items()
                        }
                    )
                    row.update(
                        {
                            f"triceps_{k}": v
                            for k, v in fft_bin_features(
                                t,
                                fs_hz=fs_hz,
                                fmin_hz=fft_fmin_hz,
                                fmax_hz=fft_fmax_hz,
                                bin_hz=fft_bin_hz,
                                log_power=fft_log_power,
                            ).items()
                        }
                    )
                feature_rows.append(row)

            # Update carry
            if tail_keep > 0 and nbuf > 0:
                carry = buf[-tail_keep:, :]
                carry_labels = buf_labels[-tail_keep:]
                carry_offset = buf_start_idx + (nbuf - tail_keep)
            else:
                carry = None
                carry_labels = None
                carry_offset = sample_index + nbuf

        sample_index += arr.shape[0]

    if export_features_csv and feature_rows:
        out_features = out_dir / "features.csv"
        # Only write the header once (first time the file is created).
        write_header = not out_features.exists()
        pd.DataFrame(feature_rows).to_csv(
            out_features,
            index=False,
            mode="a" if out_features.exists() else "w",
            header=write_header,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Mendeley sEMG recordings using angle hysteresis.")
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Folder with Subject_* recordings (default: ML/datasets/Mendeley/sEMG_recordings).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <input-root>/labeled).",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        choices=[2, 3],
        default=3,
        help="Number of discrete arm positions (default: 3).",
    )
    parser.add_argument(
        "--label-mode",
        choices=["hysteresis", "bins"],
        default="hysteresis",
        help="How to convert angle->label. 'bins' uses simple thresholds; 'hysteresis' uses state machine (default).",
    )
    parser.add_argument("--bin-straight-max", type=float, default=40.0, help="(bins, 3-state) straight if angle <= this.")
    parser.add_argument("--bin-mid-max", type=float, default=80.0, help="(bins, 3-state) mid if angle <= this.")
    parser.add_argument("--low-enter", type=float, default=20.0)
    parser.add_argument("--low-exit", type=float, default=30.0)
    parser.add_argument("--high-enter", type=float, default=120.0)
    parser.add_argument("--high-exit", type=float, default=110.0)
    parser.add_argument("--t01-up", type=float, default=50.0)
    parser.add_argument("--t10-down", type=float, default=30.0)
    parser.add_argument("--t12-up", type=float, default=110.0)
    parser.add_argument("--t21-down", type=float, default=100.0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--features", action="store_true", help="Also export per-window features to <out-dir>/features.csv")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete any existing <out-dir>/features.csv before writing new features.",
    )
    parser.add_argument("--fft", action="store_true", help="Also export FFT bin power features per window.")
    parser.add_argument("--fft-fmin", type=float, default=20.0, help="FFT feature min frequency (Hz). Default: 20.")
    parser.add_argument("--fft-fmax", type=float, default=450.0, help="FFT feature max frequency (Hz). Default: 450.")
    parser.add_argument(
        "--fft-bin-hz",
        type=float,
        default=10.0,
        help="FFT feature bin width (Hz). Default: 10. (With 200ms windows, raw df=5Hz.)",
    )
    parser.add_argument("--fft-linear", action="store_true", help="Use linear power instead of log10 power for FFT features.")
    parser.add_argument(
        "--min-label-frac",
        type=float,
        default=0.0,
        help=(
            "When exporting window features, require this fraction of samples in the window to share the same label. "
            "Set e.g. 0.8 to drop transition windows. Default 0 keeps all windows."
        ),
    )
    parser.add_argument(
        "--target",
        choices=["class", "velocity", "angle"],
        default="class",
        help="What to export as the training target: discrete class label, angle velocity, or angle (default: class).",
    )
    parser.add_argument("--vel-max-dps", type=float, default=300.0, help="Velocity normalization max (deg/s) for vel_norm.")
    parser.add_argument("--vel-edge-ms", type=float, default=20.0, help="Edge averaging window (ms) for velocity estimate.")
    parser.add_argument(
        "--vel-method",
        choices=["edge", "slope"],
        default="edge",
        help="Velocity estimation method: edge (start/end) or slope (line fit). Default: edge.",
    )
    parser.add_argument(
        "--min-abs-vel-norm",
        type=float,
        default=0.0,
        help="When --target velocity, drop windows with |vel_norm| below this threshold (default: 0).",
    )
    parser.add_argument("--angle-max-deg", type=float, default=150.0, help="When --target angle, normalize by this max angle.")
    parser.add_argument(
        "--angle-stat",
        choices=["mean", "end"],
        default="mean",
        help="When --target angle, use mean angle over window or the end sample (default: mean).",
    )
    parser.add_argument("--fs", type=float, default=2000.0)
    parser.add_argument("--win-sec", type=float, default=0.2)
    parser.add_argument("--step-sec", type=float, default=0.05)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    default_root = repo_root / "ML" / "datasets" / "Mendeley" / "sEMG_recordings"
    input_root = Path(args.input_root) if args.input_root else default_root
    if not input_root.is_absolute() and not input_root.exists():
        input_root = repo_root / input_root
    if not input_root.exists():
        raise SystemExit(f"Input root not found: {input_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (input_root / "labeled")
    if not out_dir.is_absolute() and not out_dir.exists():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.features:
        out_features = out_dir / "features.csv"
        if args.overwrite and out_features.exists():
            out_features.unlink()

    n_states = int(args.n_states)
    h2: Hysteresis | None = None
    h3: ThreeStateHysteresis | None = None
    if n_states == 2:
        h2 = Hysteresis(
            low_enter=float(args.low_enter),
            low_exit=float(args.low_exit),
            high_enter=float(args.high_enter),
            high_exit=float(args.high_exit),
        )
        h2.validate()
    else:
        if args.label_mode == "hysteresis":
            h3 = ThreeStateHysteresis(
                t01_up=float(args.t01_up),
                t10_down=float(args.t10_down),
                t12_up=float(args.t12_up),
                t21_down=float(args.t21_down),
            )
            h3.validate()
        else:
            # bins mode doesn't require hysteresis params
            h3 = None

    files = find_recording_files(input_root)
    if not files:
        raise SystemExit(f"No CSV files found under: {input_root}")

    for csv_path in files:
        label_one_file(
            csv_path,
            out_dir,
            n_states=n_states,
            h2=h2,
            h3=h3,
            label_mode=str(args.label_mode),
            bin_straight_max=float(args.bin_straight_max),
            bin_mid_max=float(args.bin_mid_max),
            chunksize=int(args.chunksize),
            export_labeled_csv=True,
            export_features_csv=bool(args.features),
            fs_hz=float(args.fs),
            win_sec=float(args.win_sec),
            step_sec=float(args.step_sec),
            export_fft=bool(args.fft),
            fft_fmin_hz=float(args.fft_fmin),
            fft_fmax_hz=float(args.fft_fmax),
            fft_bin_hz=float(args.fft_bin_hz),
            fft_log_power=not bool(args.fft_linear),
            min_label_frac=float(args.min_label_frac),
            target_mode=str(args.target),
            vel_max_dps=float(args.vel_max_dps),
            vel_edge_ms=float(args.vel_edge_ms),
            vel_method=str(args.vel_method),
            min_abs_vel_norm=float(args.min_abs_vel_norm),
            angle_max_deg=float(args.angle_max_deg),
            angle_stat=str(args.angle_stat),
        )

    print(f"Done. Labeled data written to: {out_dir}")
    if args.features:
        print(f"Features file: {out_dir / 'features.csv'}")


if __name__ == "__main__":
    main()

