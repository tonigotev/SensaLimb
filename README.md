# SensaLimb

Smart arm prosthesis project.

## ML Model (Latest Recommended): Toro Ossaba Conv1D

Use this model family as the current default:

- Model family folder: `ML/models/conv1d_angle_toro_ossaba`
- Dataset folder (required): `ML/datasets/Toro Ossaba`
- Training script: `ML/train/train_conv1d_angle_toro_ossaba.py`

## 1) Install ML dependencies

From the repository root (PowerShell):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

## 2) Download and extract the Toro Ossaba dataset (required)

Dataset source:

- https://zenodo.org/records/7946782

Extract location must be exactly:

- `ML/datasets/Toro Ossaba`

Important extraction rule:

- After extraction, your `.txt` dataset files must be inside `ML/datasets/Toro Ossaba/...` (subject subfolders).
- Do **not** leave the files nested one level too deep (for example, avoid `ML/datasets/Toro Ossaba/Toro Ossaba/...`).

Quick check (PowerShell):

```powershell
Get-ChildItem "ML/datasets/Toro Ossaba" -Recurse -Filter *.txt | Select-Object -First 5 FullName
```

## 3) Install/extract the Toro Ossaba trained model files

If you received model files as a zip/archive, extract them into:

- `ML/models/conv1d_angle_toro_ossaba`

Expected model artifacts in the selected run folder:

- `model.keras`
- `model.tflite`
- `model_int8.tflite` (optional but recommended for embedded use)
- `meta.json`

Example expected path pattern:

- `ML/models/conv1d_angle_toro_ossaba/<run_name>/model.tflite`

If your extraction creates an extra folder level, move files so that the run folder directly contains `meta.json` and model files.

## 4) Run the model (inference/validation)

Pick your run folder (example):

```powershell
$MODEL_DIR = "ML/models/conv1d_angle_toro_ossaba/seed9_20260117_034451_884"
```

### A. Compare float vs int8 model outputs

```powershell
python ML/tools/compare_tflite_models.py --model-dir "$MODEL_DIR" --input-root "ML/datasets/Toro Ossaba" --samples 50
```

### B. Generate prediction-vs-true plots

```powershell
python ML/tools/plot_tflite_pred_vs_true.py --model-dir "$MODEL_DIR" --input-root "ML/datasets/Toro Ossaba" --samples 2000
```

Output images are saved in the same run folder:

- `pred_vs_true_float.png`
- `pred_vs_true_int8.png`

## 5) (Optional) Train a new Toro Ossaba model run

```powershell
python ML/train/train_conv1d_angle_toro_ossaba.py --use-file-split --movement all --emg-source filtered --angle-history-sec 0.1 --tflite-int8 --seed 9 --run-name seed9_new
```

This creates a new run folder under `ML/models/conv1d_angle_toro_ossaba`.

## Common setup mistakes

- Dataset extracted to wrong folder (must be `ML/datasets/Toro Ossaba`).
- Extra nested folder after extraction.
- Model archive extracted to wrong location.
- Missing `model.tflite` / `meta.json` in selected run folder.
