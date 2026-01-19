from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_recording_files(root: Path) -> list[Path]:
    return sorted(root.glob("Subject_*/*.csv"))


def load_subject_max(subject_max_json: Path) -> dict:
    return json.loads(subject_max_json.read_text(encoding="utf-8"))


def infer_subject_name(csv_path: Path) -> str:
    # expects .../Subject_X/<file>.csv
    return csv_path.parent.name


def resolve_csv_path(data_root: Path, file_arg: str) -> Path:
    candidate = Path(file_arg)
    repo_root = data_root.parent.parent.parent  # <repo>/ML/datasets/Mendeley/sEMG_recordings

    probe_paths: list[Path] = []
    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        probe_paths.append(candidate)  # cwd-relative
        probe_paths.append(repo_root / candidate)  # repo-root-relative (handles ML/...)
        probe_paths.append(data_root / candidate)  # data-root-relative

    for p in probe_paths:
        if p.is_file():
            return p

    # fallback: match by filename across subjects
    # Avoid globbing with filenames containing '[' / ']' (glob character classes).
    matches = [p for p in find_recording_files(data_root) if p.name == candidate.name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]

    raise SystemExit(f"Could not find CSV file for: {file_arg}")


def normalize_one_csv(
    csv_path: Path,
    out_path: Path,
    maxima: dict[str, float],
    *,
    chunksize: int = 200_000,
    normalize_angle: bool = False,
) -> None:
    """
    Normalize columns [biceps, triceps, angle] by provided maxima and write a CSV.
    Writes incrementally in chunks to support large files.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b_max = float(maxima["biceps"])
    t_max = float(maxima["triceps"])
    a_max = float(maxima["angle"])
    if b_max <= 0 or t_max <= 0 or a_max <= 0:
        raise SystemExit(f"Invalid maxima for normalization: {maxima}")

    # Read as comma-separated (matches this dataset), without headers.
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

    first = True
    for df in reader:
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            continue

        df.iloc[:, 0] = df.iloc[:, 0] / b_max
        df.iloc[:, 1] = df.iloc[:, 1] / t_max
        # IMPORTANT: Keep angle in degrees by default (needed for labeling / thresholds).
        # Only normalize angle if explicitly requested.
        if normalize_angle:
            df.iloc[:, 2] = df.iloc[:, 2] / a_max

        df.to_csv(out_path, index=False, header=False, mode="w" if first else "a")
        first = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Mendeley sEMG recordings using subject_max.json.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Normalize a single CSV (absolute, repo-relative, or relative to sEMG_recordings).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Normalize all recordings under sEMG_recordings.",
    )
    parser.add_argument(
        "--mode",
        choices=["per-subject", "global"],
        default="per-subject",
        help="Which maxima to use for normalization (default: per-subject).",
    )
    parser.add_argument(
        "--subject-max-json",
        type=str,
        default=None,
        help="Path to subject_max.json (default: <sEMG_recordings>/plots/subject_max.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <sEMG_recordings>/normalized).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size (rows) for CSV streaming (default: 200000).",
    )
    parser.add_argument(
        "--normalize-angle",
        action="store_true",
        help="Also normalize the angle column. Default keeps angle in degrees (recommended).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent / "datasets" / "Mendeley" / "sEMG_recordings"
    if not data_root.exists():
        raise SystemExit(f"Data directory not found: {data_root}")

    subject_max_json = Path(args.subject_max_json) if args.subject_max_json else (data_root / "plots" / "subject_max.json")
    if not subject_max_json.is_file():
        raise SystemExit(
            f"subject_max.json not found: {subject_max_json}. Run: python ML/plotters/plot_mendeley_semg.py --subject-max"
        )

    stats = load_subject_max(subject_max_json)
    per_subject: dict[str, dict[str, float]] = stats["per_subject"]
    global_max: dict[str, float] = stats["global"]

    out_dir = Path(args.out_dir) if args.out_dir else (data_root / "normalized")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.all and not args.file:
        raise SystemExit("Provide either --file or --all.")

    if args.all:
        files = find_recording_files(data_root)
    else:
        files = [resolve_csv_path(data_root, args.file)]

    for csv_path in files:
        subj = infer_subject_name(csv_path)
        maxima = global_max if args.mode == "global" else per_subject.get(subj)
        if maxima is None:
            raise SystemExit(f"No maxima found for subject '{subj}'.")

        rel_subj = subj.lower()
        out_path = out_dir / rel_subj / f"{csv_path.stem}_normalized.csv"
        normalize_one_csv(
            csv_path,
            out_path,
            maxima,
            chunksize=int(args.chunksize),
            normalize_angle=bool(args.normalize_angle),
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

