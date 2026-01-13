from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RATE_HZ: float = 2000.0  # 2 kHz


def find_recording_files(root: Path) -> List[Path]:
    """
    Recursively find all CSV files under Subject_* folders.
    """
    pattern = "Subject_*/*.csv"
    return sorted(root.glob(pattern))


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


def subject_color(subject_folder_name: str) -> str | None:
    """
    Map subject folder (e.g., 'Subject_1') to a consistent color.
    """
    palette = {
        "Subject_1": "#1f77b4",
        "Subject_2": "#ff7f0e",
        "Subject_3": "#2ca02c",
        "Subject_4": "#d62728",
        "Subject_5": "#9467bd",
    }
    return palette.get(subject_folder_name)


def plot_signals(files: Iterable[Path], output_dir: Path) -> None:
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
    out_bi = output_dir / "biceps_vs_time.png"
    out_tr = output_dir / "triceps_vs_time.png"
    out_an = output_dir / "angle_vs_time.png"

    fig_bi.savefig(out_bi, dpi=150)
    fig_tr.savefig(out_tr, dpi=150)
    fig_an.savefig(out_an, dpi=150)

    print(f"Saved: {out_bi}")
    print(f"Saved: {out_tr}")
    print(f"Saved: {out_an}")

    # Also display the plots interactively
    plt.show()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    # Adjusted for new script location under ML/plotters
    data_root = script_dir.parent / "datasets" / "Mendeley" / "sEMG_recordings"
    if not data_root.exists():
        raise SystemExit(f"Data directory not found: {data_root}")

    csv_files = find_recording_files(data_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {data_root}")

    output_dir = data_root / "plots"
    plot_signals(csv_files, output_dir)


if __name__ == "__main__":
    main()

