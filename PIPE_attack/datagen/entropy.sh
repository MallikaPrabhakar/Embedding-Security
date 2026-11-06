#!/usr/bin/env bash
# Recovered-secret entropy for every PIPE FE scheme, in ONE driver. Each scheme's
# baseline + elimination variants run in PARALLEL (each streaming to its own log
# under entropy_results/logs/), and we `wait` for all to finish.
#
#   facialfe, facialfe_randomised, facialfe_e8 : HARD thresholding (exact 21/21 block
#       match) over the lattice secure-sketch codeword  -> entropy_estimate_hard_threshold.py
#   ourfe (our defense)                        : SOFT thresholding (>=4/21 block match)
#       over the native keyed-hash sketch               -> entropy_estimate_soft_threshold.py
#
# FULL POPULATION (SAMPLE_COUNT=0, ~202k probes), seed 111 over 10 enrolments,
# entropy saved PER ENROLMENT (one row per draw). Per-enrolment CSVs land in
# entropy_results/facenet_<scheme>_entropy[_elim]_seed<seed>.csv (checkpointed/resumable;
# delete stale CSVs before re-running with changed code).
set -uo pipefail
cd "$(dirname "$0")"                    # run from PIPE_attack/datagen (relative data paths)

SAMPLE_COUNT=0                          # 0 = full population
SEED=111                                # fresh _seed111 CSVs; draws = 111..120
ENROLL=10

FACIALFE_LEECH_SCALE=1461.76           # datagen_facialfe_leech.py / _randomised.py
E8_SCALE=7.25                          # datagen_facialfe_e8 facenet inverse-scale t (embedding /t)
OURFE_SCALE=144.64                     # datagen_ourdef FACENET operating scale
HARD=21                                # exact: every block must match
OURFE_SOFT=4                           # soft threshold for ourfe (>=4/21 blocks)

LOGDIR=entropy_results/logs
mkdir -p "$LOGDIR"

# Launch one invocation in the background, streaming to its own log so the parallel
# tqdm bars don't interleave into an unreadable mess.  run <tag> <script> <args...>
run() {
    local tag="$1" script="$2"; shift 2
    echo "=== launch [$tag]: $script $*  (samples=$SAMPLE_COUNT seed=$SEED enroll=$ENROLL) -> $LOGDIR/entropy_${tag}_seed${SEED}.log ==="
    python "$script" "$@" \
        --samples "$SAMPLE_COUNT" --seed "$SEED" --enrollments "$ENROLL" --per-enrollment \
        > "$LOGDIR/entropy_${tag}_seed${SEED}.log" 2>&1 &
}

# facialfe + facialfe_randomised -- HARD (exact 21/21)
run facialfe_hard         entropy_estimate_hard_threshold.py --schemes facialfe facialfe_randomised --scale "$FACIALFE_LEECH_SCALE" --blocks-to-match "$HARD"
run facialfe_hard_elim    entropy_estimate_hard_threshold.py --schemes facialfe facialfe_randomised --scale "$FACIALFE_LEECH_SCALE" --blocks-to-match "$HARD" --elimination

# facialfe_e8 -- HARD (exact)
run facialfe_e8_hard      entropy_estimate_hard_threshold.py --schemes facialfe_e8 --scale "$E8_SCALE" --blocks-to-match "$HARD"
run facialfe_e8_hard_elim entropy_estimate_hard_threshold.py --schemes facialfe_e8 --scale "$E8_SCALE" --blocks-to-match "$HARD" --elimination

# ourfe -- SOFT (>=4/21) over the native keyed-hash sketch
run ourfe_soft            entropy_estimate_soft_threshold.py --scale "$OURFE_SCALE" --blocks-to-match "$OURFE_SOFT"
run ourfe_soft_elim       entropy_estimate_soft_threshold.py --scale "$OURFE_SCALE" --blocks-to-match "$OURFE_SOFT" --elimination

echo "=== all jobs launched; waiting (tail -f $LOGDIR/entropy_*_seed${SEED}.log to watch) ==="
wait
echo; echo "=== done; per-enrolment CSVs in entropy_results/*_seed${SEED}.csv, logs in $LOGDIR/ ==="
