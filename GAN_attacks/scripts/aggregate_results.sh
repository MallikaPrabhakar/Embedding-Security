#!/bin/bash
# Score every scheme's GMI/DMI reconstructions (embedding ASR) and write
# GAN_attacks/RESULTS_SUMMARY.md. Run after the per-scheme pipelines finish.
#   Usage: aggregate_results.sh [gpu]
# Set PY to your interpreter (default: active `python`, i.e. `conda activate ddpm` first).
set -e
G=${1:-0}
PY="${PY:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 "$PY" gather_results.py
echo "wrote $ROOT/RESULTS_SUMMARY.md"
