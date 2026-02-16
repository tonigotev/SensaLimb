@echo off
setlocal

REM Run from repo root (this file's directory)
cd /d "%~dp0"

REM Generate per-record plots for ALL normalized recordings.
REM You can pass extra args, e.g.:
REM   run_plot_normalized.bat --psd --max-freq 500
python ML\plotters\plot_mendeley_semg.py --input-root "ML\datasets\Mendeley\sEMG_recordings\normalized" --per-record %*

endlocal
