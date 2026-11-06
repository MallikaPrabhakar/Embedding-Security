#!/bin/bash
# Random-guessing baselines (L2 + Rep) for every scheme on the TEST split of every dataset,
# accumulated into random_guessing_results.csv (mean +/- std over enrollment images).
#
#   bash run_random_guessing.sh
#   JOBS=8 REP_SUBSET=20 SCHEMES="ourfe facialfe_e8 randomfe" DATASETS=celeba \
#       bash run_random_guessing.sh                          # throttled / smoke run
#
# All schemes are pure numpy (randomfe's Rep uses the saved LLL basis, no sage). CPU-only.
# Random guessing always uses the TEST split for every dataset. REP_SUBSET is the leave-one-out
# image subset for BOTH the L2 and Rep baselines (each image authenticates as every other ->
# N*(N-1) attempts; the 1000-image test set -> 999000). Each is O(N^2) and single-threaded, so
# the (scheme x dataset) jobs run IN PARALLEL: each writes its own temp CSV (no concurrent-write
# races) and they are merged (upserted by dataset/model/scheme/folders) into RG_CSV at the end.
# JOBS caps concurrency (default = all jobs at once); the bottleneck scheme (e8 / leech) is
# ~25-30 min per dataset, so the whole sweep finishes in roughly that time, not the sum.
cd "$(dirname "$0")" || exit 1                 # -> PIPE_attack/
HERE=$(pwd)

PY=${PY:-$HOME/miniconda3/envs/ddpm/bin/python}
MODEL=${MODEL:-facenet}
REP_SUBSET=${REP_SUBSET:-1000}
SCHEMES=${SCHEMES:-"unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe randomfe"}
DATASETS=${DATASETS:-"celeba casia_webface lfw"}
RG_CSV=${RG_CSV:-"$HERE/random_guessing_results.csv"}
JOBS=${JOBS:-0}                                # max concurrent jobs (0 = no cap, launch all)

LOGDIR="$HERE/run_experiments/logs"; mkdir -p "$LOGDIR"
TMP=$(mktemp -d)

echo "=== random guessing (TEST split, PARALLEL) : schemes=[$SCHEMES] datasets=[$DATASETS] model=$MODEL"
echo "    rep_subset=$REP_SUBSET  jobs=$([ "$JOBS" -gt 0 ] && echo "$JOBS" || echo all)  py=$PY"
echo "    per-job temp CSVs in $TMP  ->  merged into $RG_CSV"

run_job() {  # scheme dataset -> own temp CSV (cwd=datagen so ../datasets resolves)
    local sc=$1 ds=$2
    ( cd "$HERE/datagen" && $PY ../random_guessing.py \
        --dataset "$ds" --model "$MODEL" --scheme "$sc" \
        --folders test --rep_subset "$REP_SUBSET" --csv "$TMP/${sc}_${ds}.csv" ) \
        > "$LOGDIR/rg_${sc}_${ds}.log" 2>&1
}

pids=()
for sc in $SCHEMES; do
    for ds in $DATASETS; do
        run_job "$sc" "$ds" & pids+=("$!")
        echo "  launched $sc/$ds (pid $!)"
        if [ "$JOBS" -gt 0 ]; then
            while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        fi
    done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "==== all jobs finished (failures=$fail) ; per-scheme tail ===="
for sc in $SCHEMES; do for ds in $DATASETS; do
    echo "  -- $sc/$ds --"; grep -E "\[L2\]|\[Rep\] scheme=.*ASR|Traceback|Error" "$LOGDIR/rg_${sc}_${ds}.log" 2>/dev/null | tail -2
done; done

# merge per-job temp CSVs into RG_CSV (upsert by dataset/model/scheme/folders)
echo; echo "==== merging $(ls "$TMP"/*.csv 2>/dev/null | wc -l) result files -> $RG_CSV ===="
$PY - "$RG_CSV" "$TMP"/*.csv <<'PYEOF'
import sys, os, pandas as pd
out, parts = sys.argv[1], [p for p in sys.argv[2:] if os.path.exists(p) and os.path.getsize(p) > 0]
key = ["dataset", "model", "scheme", "folders"]
frames = ([pd.read_csv(out)] if os.path.exists(out) else []) + [pd.read_csv(p) for p in parts]
if not frames:
    print("[merge] no result files"); sys.exit(0)
df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=key, keep="last")
df = df.sort_values(key).reset_index(drop=True)
df.to_csv(out, index=False)
print(f"[merge] {len(df)} rows -> {out}")
PYEOF

echo; echo "==== done -> $RG_CSV ===="; [ -f "$RG_CSV" ] && cat "$RG_CSV"
