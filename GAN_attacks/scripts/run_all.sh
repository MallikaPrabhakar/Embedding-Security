#!/bin/bash
# Reproduce ALL of GAN_attacks/RESULTS_SUMMARY.md: for each scheme run the
# classifier-training -> GMI -> DMI pipeline (one GPU per scheme, in parallel),
# then aggregate the embedding ASR table.
#
# Schemes (5): unprotected, facialfe_leech, facialfe_e8, facialfe_leech_randomised, ourfe.
# Each pipeline is CPU-FE bound, so 4 running at once share CPU cores -- expect a long run.
# Edit the GPU indices to match your free devices, or run schemes one at a time with
# run_pipeline.sh. Set PY to your interpreter (default: the active `python`, i.e. run
# `conda activate ddpm` first).
#   Usage: bash run_all.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"

bash "$HERE/run_pipeline.sh" unprotected               0 > "$HERE/logs/pipe_unprotected.log" 2>&1 &
bash "$HERE/run_pipeline.sh" facialfe_leech            1 > "$HERE/logs/pipe_facialfe_leech.log" 2>&1 &
bash "$HERE/run_pipeline.sh" facialfe_e8               2 > "$HERE/logs/pipe_facialfe_e8.log" 2>&1 &
bash "$HERE/run_pipeline.sh" facialfe_leech_randomised 3 > "$HERE/logs/pipe_randomised.log" 2>&1 &
bash "$HERE/run_pipeline.sh" ourfe                     4 > "$HERE/logs/pipe_ourfe.log" 2>&1 &
wait

echo "=== all pipelines done; aggregating -> RESULTS_SUMMARY.md ==="
bash "$HERE/aggregate_results.sh" 0
