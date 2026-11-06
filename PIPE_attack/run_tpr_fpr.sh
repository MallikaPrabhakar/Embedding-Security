#!/bin/bash
# TPR/FPR verification for every scheme x dataset, accumulated into tpr_fpr_results.csv
# (TPR on genuine pairs, FPR on impostor pairs, via tpr_fpr.py + helper.build_pairs).
#
#   bash run_tpr_fpr.sh
#   PAIRS=25 SCHEMES="ourfe facialfe_e8 randomfe" DATASETS=celeba \
#       bash run_tpr_fpr.sh                                   # quick smoke run
#
# All schemes are pure numpy (randomfe uses the saved LLL basis, no sage). CPU-only.
# Per dataset the split is fixed: celeba -> test (test+test2 pairs), casia_webface/lfw -> train
# (train+test_seen pairs). PAIRS = genuine AND impostor pairs EACH; build_pairs uses samples//2
# per class, so we pass --samples 2*PAIRS. The CSV is upserted per (dataset,model,mode,scheme)
# and the loop is sequential, so writes never race.
cd "$(dirname "$0")" || exit 1                 # -> PIPE_attack/
HERE=$(pwd)

PY=${PY:-$HOME/miniconda3/envs/ddpm/bin/python}
MODEL=${MODEL:-facenet}
PAIRS=${PAIRS:-1000}                           # genuine pairs AND impostor pairs each
SCHEMES=${SCHEMES:-"unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe randomfe"}
DATASETS=${DATASETS:-"celeba casia_webface lfw"}
TF_CSV=${TF_CSV:-"$HERE/tpr_fpr_results.csv"}

declare -A SPLIT=( [celeba]=test [casia_webface]=train [lfw]=train )

LOGDIR="$HERE/run_experiments/logs"; mkdir -p "$LOGDIR"

echo "=== tpr/fpr : schemes=[$SCHEMES] datasets=[$DATASETS] model=$MODEL"
echo "    pairs=$PAIRS/class (samples=$((PAIRS*2)))  py=$PY  -> $TF_CSV"

for sc in $SCHEMES; do
    for ds in $DATASETS; do
        split=${SPLIT[$ds]}
        if [ -z "$split" ]; then echo "skip $sc/$ds: unknown dataset (no split mapping)"; continue; fi
        echo; echo "==== $sc / $ds (mode=$split) ===="
        # tpr_fpr.py self-chdirs into datagen, so it is runnable from anywhere
        $PY "$HERE/tpr_fpr.py" \
            --scheme "$sc" --dataset "$ds" --model "$MODEL" \
            --mode "$split" --samples "$((PAIRS*2))" --csv "$TF_CSV" 2>&1 | tee "$LOGDIR/tf_${sc}_${ds}.log"
    done
done

echo; echo "==== done -> $TF_CSV ===="; [ -f "$TF_CSV" ] && cat "$TF_CSV"
