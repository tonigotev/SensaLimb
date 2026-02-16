from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, welch


SAMPLE_RATE_HZ: float = 2000.0  # 2 kHz


def find_recording_files(root: Path) -> List[Path]:
    """
    Recursively find all CSV files under Subject_* or subject_* folders.
    """
    files = list(root.glob("Subject_*/*.csv")) + list(root.glob("subject_*/*.csv"))
    return sorted(set(files))


def iter_subject_dirs(root: Path) -> List[Path]:
    dirs = [p for p in root.glob("Subject_*") if p.is_dir()] + [p for p in root.glob("subject_*") if p.is_dir()]
    return sorted({d for d in dirs})


def load_three_column_csv(file_path: Path) -> np.ndarray:
    """
    Load a CSV file with 3 numeric columns.
    Uses Python engine with sep=None to auto-detect delimiters.
    Returns a (N, 3) numpy array of floats, dropping any non-numeric rows.
    """
    df = pd.read_csv(
        file_path,
        sep=None,  # auto-detect delimiter
        engine="python",
        header=None,
        usecols=[0, 1, 2],
        comment="#",
        on_bad_lines="skip",  # skip malformed rows
    )
    # Coerce to numeric and drop rows with NaNs (non-numeric after coercion)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    values = df.to_numpy(dtype=float, copy=False)
    if values.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {file_path}, got {values.shape[1]}")
    return values


def update_maxima(current: dict[str, float], chunk: np.ndarray, *, use_abs: bool) -> dict[str, float]:
    """
    Update maxima dict with values from chunk (N,3): [biceps, triceps, angle].
    """
    if chunk.size == 0:
        return current
    data = np.abs(chunk) if use_abs else chunk
    # column maxima for this chunk
    m = np.max(data, axis=0)
    current["biceps"] = max(current["biceps"], float(m[0]))
    current["triceps"] = max(current["triceps"], float(m[1]))
    current["angle"] = max(current["angle"], float(m[2]))
    return current


def compute_subject_maxima(
    data_root: Path,
    *,
    use_abs: bool = True,
    chunksize: int = 200_000,
) -> dict[str, dict[str, float]]:
    """
    Compute per-subject maxima across all recordings.

    Returns:
      {
        "Subject_1": {"biceps": ..., "triceps": ..., "angle": ...},
        ...
      }

    Notes:
    - Default uses absolute maxima (best for normalization of signed EMG).
    - Uses chunked CSV reading to avoid loading huge files fully into RAM.
    """
    results: dict[str, dict[str, float]] = {}
    for subj_dir in iter_subject_dirs(data_root):
        subj_name = subj_dir.name
        maxima = {"biceps": 0.0, "triceps": 0.0, "angle": 0.0}

        csv_files = sorted(subj_dir.glob("*.csv"))
        for csv_path in csv_files:
            # Fast path: these files are comma-separated in this dataset.
            # We still coerce to numeric and drop bad rows per chunk.
            reader = pd.read_csv(
                csv_path,
                sep=",",
                engine="c",
                header=None,
                usecols=[0, 1, 2],
                comment="#",
                on_bad_lines="skip",
                chunksize=chunksize,
            )
            for df in reader:
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                arr = df.to_numpy(dtype=float, copy=False)
                maxima = update_maxima(maxima, arr, use_abs=use_abs)

        results[subj_name] = maxima
    return results


def subject_color(subject_folder_name: str) -> str | None:
    """
    Map subject folder (e.g., 'Subject_1') to a consistent color.
    """
    subject_folder_name = subject_folder_name.replace("subject_", "Subject_")
    palette = {
        "Subject_1": "#1f77b4",
        "Subject_2": "#ff7f0e",
        "Subject_3": "#2ca02c",
        "Subject_4": "#d62728",
        "Subject_5": "#9467bd",
    }
    return palette.get(subject_folder_name)


def bandpass_filter(
    x: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Simple zero-phase Butterworth bandpass filter.
    EMG commonly uses ~20-450 Hz; adjust as needed.
    """
    if low_hz <= 0 or high_hz >= fs_hz / 2:
        raise ValueError("Bandpass cutoffs must satisfy 0 < low < high < Nyquist.")
    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz)
    return filtfilt(b, a, x, axis=0)


def notch_filter(x: np.ndarray, fs_hz: float, notch_hz: float, q: float = 30.0) -> np.ndarray:
    """
    Apply a zero-phase IIR notch filter at notch_hz.

    Useful for removing mains interference (50/60 Hz) and optionally harmonics.
    q controls notch sharpness (higher = narrower).
    """
    if notch_hz <= 0 or notch_hz >= fs_hz / 2:
        raise ValueError("notch_hz must satisfy 0 < notch_hz < Nyquist.")
    b, a = iirnotch(w0=notch_hz, Q=q, fs=fs_hz)
    return filtfilt(b, a, x, axis=0)


def welch_psd(
    x: np.ndarray,
    fs_hz: float,
    nperseg: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD estimate.
    Returns (freq_hz, psd).
    """
    f, pxx = welch(x, fs=fs_hz, nperseg=nperseg, detrend="constant", scaling="density")
    return f, pxx


def spectral_features(freq_hz: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    """
    Compute common spectral metrics.
    - peak_freq: frequency of max PSD
    - mean_freq: sum(f*P)/sum(P)
    - median_freq: frequency where cumulative power reaches 50%
    """
    power = np.maximum(psd, 0.0)
    def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
        # Prefer NumPy's implementation when present; otherwise manual trapezoid rule.
        trapezoid_fn = getattr(np, "trapezoid", None)
        if callable(trapezoid_fn):
            return float(trapezoid_fn(y, x))
        dx = np.diff(x)
        if dx.size == 0:
            return 0.0
        return float(np.sum((y[:-1] + y[1:]) * 0.5 * dx))

    total = integrate_trapezoid(power, freq_hz)
    if total <= 0:
        return {"peak_freq": float("nan"), "mean_freq": float("nan"), "median_freq": float("nan")}

    peak_freq = float(freq_hz[int(np.argmax(power))])
    mean_freq = float(integrate_trapezoid(freq_hz * power, freq_hz) / total)

    cum = np.cumsum(power)
    cum = cum / cum[-1]
    median_idx = int(np.searchsorted(cum, 0.5))
    median_freq = float(freq_hz[min(median_idx, len(freq_hz) - 1)])

    return {"peak_freq": peak_freq, "mean_freq": mean_freq, "median_freq": median_freq}


def plot_psd_for_file(
    csv_path: Path,
    output_dir: Path,
    max_freq_hz: float,
    bandpass: tuple[float, float] | None,
    *,
    notches: list[float] | None = None,
    notch_q: float = 30.0,
) -> None:
    """
    For a single CSV: compute and plot PSD for biceps and triceps,
    and print peak/mean/median frequency for each.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_three_column_csv(csv_path)
    if data.shape[0] == 0:
        raise SystemExit(f"No data rows found in: {csv_path}")

    biceps = data[:, 0]
    triceps = data[:, 1]

    if bandpass is not None:
        low, high = bandpass
        filtered = bandpass_filter(np.column_stack([biceps, triceps]), SAMPLE_RATE_HZ, low, high)
        biceps = filtered[:, 0]
        triceps = filtered[:, 1]

    if notches:
        filtered = np.column_stack([biceps, triceps])
        for hz in notches:
            filtered = notch_filter(filtered, SAMPLE_RATE_HZ, float(hz), q=float(notch_q))
        biceps = filtered[:, 0]
        triceps = filtered[:, 1]

    f_bi, p_bi = welch_psd(biceps, SAMPLE_RATE_HZ)
    f_tr, p_tr = welch_psd(triceps, SAMPLE_RATE_HZ)

    mask_bi = f_bi <= max_freq_hz
    mask_tr = f_tr <= max_freq_hz

    feats_bi = spectral_features(f_bi[mask_bi], p_bi[mask_bi])
    feats_tr = spectral_features(f_tr[mask_tr], p_tr[mask_tr])

    print(f"PSD metrics for: {csv_path}")
    print(
        "Biceps  (Hz): "
        f"peak={feats_bi['peak_freq']:.2f}, mean={feats_bi['mean_freq']:.2f}, median={feats_bi['median_freq']:.2f}"
    )
    print(
        "Triceps (Hz): "
        f"peak={feats_tr['peak_freq']:.2f}, mean={feats_tr['mean_freq']:.2f}, median={feats_tr['median_freq']:.2f}"
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(f_bi[mask_bi], 10 * np.log10(p_bi[mask_bi] + 1e-20), label="Biceps PSD")
    ax.plot(f_tr[mask_tr], 10 * np.log10(p_tr[mask_tr] + 1e-20), label="Triceps PSD")
    ax.set_title("PSD (Welch)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out = output_dir / f"{csv_path.stem}_psd.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    # In batch mode we never show; close to avoid accumulating figures.
    plt.close(fig)


def plot_time_frequency_for_file(
    csv_path: Path,
    output_dir: Path,
    *,
    max_freq_hz: float,
    bandpass: tuple[float, float] | None,
    notches: list[float] | None,
    notch_q: float,
    win_sec: float,
    step_sec: float,
    metric: str,
    vmin_db: float | None,
    vmax_db: float | None,
) -> None:
    """
    Compute a time-varying spectrum (sliding-window PSD / spectrogram) for biceps & triceps.

    - Spectrogram: Power spectral density over time, shown as dB/Hz vs Frequency
    - Metric over time: peak/mean/median frequency per window

    Typical EMG: use bandpass ~20-450 Hz, fs=2kHz.
    """
    if win_sec <= 0 or step_sec <= 0:
        raise ValueError("win_sec and step_sec must be > 0")

    data = load_three_column_csv(csv_path)
    if data.shape[0] == 0:
        raise SystemExit(f"No data rows found in: {csv_path}")

    biceps = data[:, 0]
    triceps = data[:, 1]

    if bandpass is not None:
        low, high = bandpass
        filtered = bandpass_filter(np.column_stack([biceps, triceps]), SAMPLE_RATE_HZ, low, high)
        biceps = filtered[:, 0]
        triceps = filtered[:, 1]

    # Optional notch filtering (e.g., 50/60 Hz and harmonics)
    if notches:
        filtered = np.column_stack([biceps, triceps])
        for hz in notches:
            filtered = notch_filter(filtered, SAMPLE_RATE_HZ, float(hz), q=float(notch_q))
        biceps = filtered[:, 0]
        triceps = filtered[:, 1]

    nperseg = int(round(win_sec * SAMPLE_RATE_HZ))
    nstep = int(round(step_sec * SAMPLE_RATE_HZ))
    if nperseg < 8:
        raise SystemExit(f"Window too small: win_sec={win_sec} -> nperseg={nperseg}")
    if nstep < 1:
        raise SystemExit(f"Step too small: step_sec={step_sec} -> nstep={nstep}")
    noverlap = nperseg - nstep
    if noverlap < 0:
        noverlap = 0

    def compute_spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, t, sxx = spectrogram(
            x,
            fs=SAMPLE_RATE_HZ,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        mask = f <= max_freq_hz
        return f[mask], t, sxx[mask, :]

    f_bi, t_bi, sxx_bi = compute_spec(biceps)
    f_tr, t_tr, sxx_tr = compute_spec(triceps)

    # Feature per time-slice
    def feature_over_time(freq: np.ndarray, sxx: np.ndarray) -> np.ndarray:
        # sxx: (F, T)
        if metric == "peak":
            return freq[np.argmax(sxx, axis=0)]
        if metric == "mean":
            num = np.sum(freq[:, None] * sxx, axis=0)
            den = np.sum(sxx, axis=0) + 1e-30
            return num / den
        if metric == "median":
            cs = np.cumsum(sxx, axis=0)
            cs = cs / (cs[-1, :] + 1e-30)
            idx = np.argmax(cs >= 0.5, axis=0)
            return freq[idx]
        raise SystemExit(f"Unknown metric: {metric}")

    feat_bi = feature_over_time(f_bi, sxx_bi)
    feat_tr = feature_over_time(f_tr, sxx_tr)

    # Plot spectrograms + feature curves
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    bi_db = 10 * np.log10(sxx_bi + 1e-20)
    tr_db = 10 * np.log10(sxx_tr + 1e-20)

    im0 = axes[0].pcolormesh(t_bi, f_bi, bi_db, shading="auto", vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title(f"Biceps Spectrogram (PSD, dB/Hz) — win={win_sec}s step={step_sec}s")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], label="dB/Hz")

    im1 = axes[1].pcolormesh(t_tr, f_tr, tr_db, shading="auto", vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title(f"Triceps Spectrogram (PSD, dB/Hz) — win={win_sec}s step={step_sec}s")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1], label="dB/Hz")

    axes[2].plot(t_bi, feat_bi, label=f"Biceps {metric} freq")
    axes[2].plot(t_tr, feat_tr, label=f"Triceps {metric} freq")
    axes[2].set_title(f"Frequency over time ({metric})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[2].legend()

    fig.tight_layout()
    out = output_dir / f"{csv_path.stem}_time_frequency_{metric}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_signals(
    files: Iterable[Path],
    output_dir: Path,
    *,
    show: bool = True,
    single_file: Path | None = None,
) -> None:
    """
    Create three figures:
      1) Biceps vs Time
      2) Triceps vs Time
      3) Angle vs Time
    Overlay all recordings; color by subject; save PNGs to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare three separate figures
    fig_bi, ax_bi = plt.subplots(figsize=(12, 4.5))
    fig_tr, ax_tr = plt.subplots(figsize=(12, 4.5))
    fig_an, ax_an = plt.subplots(figsize=(12, 4.5))

    for csv_path in files:
        try:
            data = load_three_column_csv(csv_path)
        except Exception as exc:
            print(f"Skipping {csv_path} due to read error: {exc}")
            continue

        num_samples = data.shape[0]
        if num_samples == 0:
            print(f"Skipping {csv_path}: no data rows found.")
            continue

        time_s = np.arange(num_samples, dtype=float) / SAMPLE_RATE_HZ

        subj_name = csv_path.parent.name
        color = subject_color(subj_name)

        # Column mapping from prompt:
        # 0: biceps, 1: triceps, 2: elbow angle
        ax_bi.plot(
            time_s,
            data[:, 0],
            linewidth=0.7,
            alpha=0.55,
            color=color,
        )
        ax_tr.plot(
            time_s,
            data[:, 1],
            linewidth=0.7,
            alpha=0.55,
            color=color,
        )
        ax_an.plot(
            time_s,
            data[:, 2],
            linewidth=0.7,
            alpha=0.55,
            color=color,
        )

    # Label and style axes
    ax_bi.set_title("Biceps sEMG vs Time (All Recordings)")
    ax_bi.set_xlabel("Time (s)")
    ax_bi.set_ylabel("Amplitude")
    ax_bi.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax_tr.set_title("Triceps sEMG vs Time (All Recordings)")
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Amplitude")
    ax_tr.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax_an.set_title("Elbow Angle vs Time (All Recordings)")
    ax_an.set_xlabel("Time (s)")
    ax_an.set_ylabel("Angle (deg)")
    ax_an.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig_bi.tight_layout()
    fig_tr.tight_layout()
    fig_an.tight_layout()

    # Save figures
    if single_file is not None:
        stem = single_file.stem
        out_bi = output_dir / f"{stem}_biceps_plot.png"
        out_tr = output_dir / f"{stem}_triceps_plot.png"
        out_an = output_dir / f"{stem}_angle_plot.png"
    else:
        out_bi = output_dir / "biceps_vs_time.png"
        out_tr = output_dir / "triceps_vs_time.png"
        out_an = output_dir / "angle_vs_time.png"

    fig_bi.savefig(out_bi, dpi=150)
    fig_tr.savefig(out_tr, dpi=150)
    fig_an.savefig(out_an, dpi=150)

    print(f"Saved: {out_bi}")
    print(f"Saved: {out_tr}")
    print(f"Saved: {out_an}")

    # Also display the plots interactively (optionally)
    if show:
        plt.show()
    else:
        plt.close(fig_bi)
        plt.close(fig_tr)
        plt.close(fig_an)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mendeley sEMG recordings.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help=(
            "Optional path to a specific CSV to plot. "
            "Can be absolute, relative to repo root, or relative to sEMG_recordings."
        ),
    )
    parser.add_argument(
        "--psd",
        action="store_true",
        help="Also compute and plot Welch PSD for biceps/triceps (best used with --file).",
    )
    parser.add_argument(
        "--tf",
        action="store_true",
        help=(
            "Compute a time-varying spectrum (spectrogram) for biceps/triceps and save it. "
            "Best used with --file, or combined with --per-record."
        ),
    )
    parser.add_argument(
        "--tf-win",
        type=float,
        default=0.1,
        help="Time-frequency window length in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--tf-step",
        type=float,
        default=0.1,
        help="Time-frequency step size in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--tf-metric",
        choices=["peak", "mean", "median"],
        default="peak",
        help="Frequency summary to plot over time (default: peak).",
    )
    parser.add_argument(
        "--notch",
        type=str,
        default=None,
        help='Optional notch frequencies in Hz, e.g. "60" or "60,120,180" to suppress mains and harmonics.',
    )
    parser.add_argument(
        "--notch-q",
        type=float,
        default=30.0,
        help="Notch filter Q (sharpness). Higher = narrower notch. Default: 30.",
    )
    parser.add_argument(
        "--tf-vmin",
        type=float,
        default=None,
        help="Optional spectrogram color scale min (dB/Hz).",
    )
    parser.add_argument(
        "--tf-vmax",
        type=float,
        default=None,
        help="Optional spectrogram color scale max (dB/Hz).",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=500.0,
        help="Max frequency (Hz) to show/compute PSD metrics for (default: 500).",
    )
    parser.add_argument(
        "--bandpass",
        type=str,
        default=None,
        help='Optional bandpass like "20,450" (Hz) applied before PSD.',
    )
    parser.add_argument(
        "--per-record",
        action="store_true",
        help=(
            "Generate plots for each record into plots/<subject>/<record_stem>/... "
            "(no interactive windows). Ignored if --file is provided."
        ),
    )
    parser.add_argument(
        "--subject-max",
        action="store_true",
        help=(
            "Scan all recordings and compute per-subject maxima for biceps/triceps/angle. "
            "Writes JSON to plots/subject_max.json and prints a summary."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help=(
            "Folder containing recordings (default: ML/datasets/Mendeley/sEMG_recordings). "
            "To plot normalized data, set to: ML/datasets/Mendeley/sEMG_recordings/normalized"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help=(
            "Where to write plots. Default: <input-root>/plots (so normalized data goes to normalized/plots)."
        ),
    )
    parser.add_argument(
        "--signed-max",
        action="store_true",
        help="Use signed maxima instead of absolute maxima (default uses absolute maxima).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size (rows) for subject-max scan (default: 200000).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # Default input root: ML/datasets/Mendeley/sEMG_recordings
    default_root = script_dir.parent / "datasets" / "Mendeley" / "sEMG_recordings"

    if args.input_root:
        candidate_root = Path(args.input_root)
        # If user passes "ML/..." while running from inside ML/, it won't exist; try repo-root-relative.
        if not candidate_root.is_absolute() and not candidate_root.exists():
            candidate_root = repo_root / candidate_root
        data_root = candidate_root
    else:
        data_root = default_root

    if not data_root.exists():
        raise SystemExit(f"Data directory not found: {data_root}")

    if args.out_root:
        candidate_out = Path(args.out_root)
        if not candidate_out.is_absolute() and not candidate_out.exists():
            candidate_out = repo_root / candidate_out
        output_root = candidate_out
    else:
        output_root = data_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    # Stats-only mode
    if args.subject_max:
        output_dir = output_root
        per_subject = compute_subject_maxima(
            data_root,
            use_abs=not args.signed_max,
            chunksize=int(args.chunksize),
        )

        # Also compute global maxima across subjects
        global_max = {"biceps": 0.0, "triceps": 0.0, "angle": 0.0}
        for subj_name, m in per_subject.items():
            global_max["biceps"] = max(global_max["biceps"], float(m["biceps"]))
            global_max["triceps"] = max(global_max["triceps"], float(m["triceps"]))
            global_max["angle"] = max(global_max["angle"], float(m["angle"]))

        payload = {
            "mode": "abs" if not args.signed_max else "signed",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "per_subject": per_subject,
            "global": global_max,
        }
        out_json = output_dir / "subject_max.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"Saved: {out_json}")
        print("Per-subject maxima:")
        for subj_name in sorted(per_subject.keys()):
            m = per_subject[subj_name]
            print(
                f"- {subj_name}: biceps={m['biceps']:.6g}, triceps={m['triceps']:.6g}, angle={m['angle']:.6g}"
            )
        print(
            f"Global maxima: biceps={global_max['biceps']:.6g}, triceps={global_max['triceps']:.6g}, angle={global_max['angle']:.6g}"
        )
        return

    if args.file:
        if args.file.strip() in {"...", "\"...\"", "'...'"}:
            raise SystemExit(
                "You passed a placeholder '...'. Replace it with a real CSV path, e.g.\n"
                '  --file "ML\\datasets\\Mendeley\\sEMG_recordings\\Subject_2\\record-[2014.12.11-11.7.18].csv"\n'
                'or just:\n'
                '  --file "record-[2014.12.11-11.7.18].csv"'
            )
        # Resolve target CSV path robustly
        candidate = Path(args.file)
        resolved: Path | None = None

        # Try in order: as-given, repo-root-relative, data-root-relative
        probe_paths: list[Path] = []
        if candidate.is_absolute():
            probe_paths.append(candidate)
        else:
            probe_paths.append(candidate)  # relative to CWD
            probe_paths.append(repo_root / candidate)  # relative to repo root (handles 'ML/...')
            probe_paths.append(data_root / candidate)  # relative to sEMG_recordings

        for p in probe_paths:
            if p.is_file():
                resolved = p
                break

        if not resolved:
            # Try matching by filename across subjects
            name = candidate.name
            # Avoid globbing with filenames containing '[' / ']' (glob character classes).
            matches = [p for p in find_recording_files(data_root) if p.name == name]
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                print(f"Multiple matches for '{args.file}'. Using first: {matches[0]}")
                resolved = matches[0]

        if not resolved or not resolved.is_file():
            raise SystemExit(f"Could not find CSV file for: {args.file}")

        csv_files = [resolved]
        print(f"Plotting single file: {resolved}")
    else:
        csv_files = find_recording_files(data_root)

    if not csv_files:
        raise SystemExit(f"No CSV files found under {data_root}")

    # Output folder:
    # - default: <out-root> (aggregated)
    # - with --file: <out-root>/<subject>/<record_stem>/ (per-recording)
    output_dir = output_root
    single_output_dir: Path | None = None
    if args.file and csv_files:
        subj_dir = csv_files[0].parent.name.lower()
        single_output_dir = output_dir / subj_dir / csv_files[0].stem
        single_output_dir.mkdir(parents=True, exist_ok=True)

    def parse_bandpass() -> tuple[float, float] | None:
        if not args.bandpass:
            return None
        parts = [p.strip() for p in args.bandpass.split(",")]
        if len(parts) != 2:
            raise SystemExit('Invalid --bandpass. Expected format like "20,450".')
        return (float(parts[0]), float(parts[1]))

    def parse_notches() -> list[float]:
        if not args.notch:
            return []
        parts = [p.strip() for p in args.notch.split(",") if p.strip()]
        return [float(p) for p in parts]

    # Batch mode: generate per-record outputs for all recordings (no GUI windows).
    if args.per_record and not args.file:
        bp = parse_bandpass()
        notches = parse_notches()
        for csv_path in csv_files:
            subj_dir = csv_path.parent.name.lower()
            per_dir = output_dir / subj_dir / csv_path.stem
            per_dir.mkdir(parents=True, exist_ok=True)
            plot_signals([csv_path], per_dir, show=False, single_file=csv_path)
            if args.psd:
                plot_psd_for_file(
                    csv_path,
                    per_dir,
                    max_freq_hz=float(args.max_freq),
                    bandpass=bp,
                    notches=notches,
                    notch_q=float(args.notch_q),
                )
            if args.tf:
                plot_time_frequency_for_file(
                    csv_path,
                    per_dir,
                    max_freq_hz=float(args.max_freq),
                    bandpass=bp,
                    notches=notches,
                    notch_q=float(args.notch_q),
                    win_sec=float(args.tf_win),
                    step_sec=float(args.tf_step),
                    metric=str(args.tf_metric),
                    vmin_db=args.tf_vmin,
                    vmax_db=args.tf_vmax,
                )
        print(f"Done. Per-record plots saved under: {output_dir}")
        return

    # Non-batch: If we're also doing PSD, create all figures first and show once at the end,
    # so the initial plt.show() doesn't block PSD computation.
    plot_signals(
        csv_files,
        single_output_dir or output_dir,
        show=not args.psd,
        single_file=csv_files[0] if args.file and csv_files else None,
    )

    if args.psd:
        bp = parse_bandpass()
        notches = parse_notches()
        # If multiple files were plotted, default PSD to the first one to avoid a cluttered plot.
        target = csv_files[0]
        plot_psd_for_file(
            target,
            single_output_dir or output_dir,
            max_freq_hz=float(args.max_freq),
            bandpass=bp,
            notches=notches,
            notch_q=float(args.notch_q),
        )
        plt.show()

    if args.tf:
        bp = parse_bandpass()
        notches = parse_notches()
        target = csv_files[0]
        plot_time_frequency_for_file(
            target,
            single_output_dir or output_dir,
            max_freq_hz=float(args.max_freq),
            bandpass=bp,
            notches=notches,
            notch_q=float(args.notch_q),
            win_sec=float(args.tf_win),
            step_sec=float(args.tf_step),
            metric=str(args.tf_metric),
            vmin_db=args.tf_vmin,
            vmax_db=args.tf_vmax,
        )


if __name__ == "__main__":
    main()

