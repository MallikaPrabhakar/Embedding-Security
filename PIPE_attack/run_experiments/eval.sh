#!/bin/bash
# Score the reconstructions from sample.sh for each scheme x dataset (in-domain celeba +
# cross-dataset lfw / casia_webface). Consolidates the old evaluate*.sh and the eval halves
# of eval_crossdataset*.sh. Two metrics, both in the deepface env (Facenet512):
#   L2  ASR : scripts/evaluation.py     -> <res>/test_distances.npy            (euclidean_l2 < 1.1027)
#   Rep ASR : scripts/eval_protected.py -> <res>/test_protected_blocks_<tag>.npy (recovered-secret;
#             protected schemes only -- unprotected has no FE secret, L2 is its only metric)
#
#   Usage:  [SCHEMES="..."] [DATASETS="celeba lfw casia_webface"] [GPUS="0 1 2 3"] [NUM=1000] \
#           bash run_experiments/eval.sh
#   Run from PIPE_attack/ AFTER sample.sh.
if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base); else echo "conda not found"; exit 1; fi
cd "$(dirname "$0")/.." || exit 1          # -> PIPE_attack/
source "$CONDA_BASE/bin/activate" deepface

SCHEMES=${SCHEMES:-"unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe"}
DATASETS=${DATASETS:-"celeba lfw casia_webface"}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
NUM=${NUM:-1000}
NPZ="test_samples_${NUM}x64x64x3.npz"
LOGDIR=run_experiments/logs; mkdir -p "$LOGDIR"

# checkpoint per scheme (must match sample.sh) and which schemes get the rep-ASR pass
declare -A CKPT=(    [unprotected]=025000 [facialfe_e8]=025000 [facialfe_leech]=050000 [facialfe_leech_randomised]=050000 [ourfe]=050000 )
declare -A IS_PROT=( [facialfe_e8]=1      [facialfe_leech]=1   [facialfe_leech_randomised]=1 [ourfe]=1 )

res_dir() {  # scheme dataset ckpt -> result dir (same layout as sample.sh)
    if [ "$2" = celeba ]; then echo "results/ckpt_$3/$1/celeba/facenet"
    else echo "results/crossdataset/$2/ckpt_$3/$1/celeba/facenet"; fi
}

i=0
for s in $SCHEMES; do
    if [ -z "${CKPT[$s]}" ]; then echo "skip $s: unknown scheme"; continue; fi
    for ds in $DATASETS; do
        g=${GPUS[$((i % ${#GPUS[@]}))]}; ck=${CKPT[$s]}
        out=$(res_dir "$s" "$ds" "$ck")
        if [ ! -f "$out/$NPZ" ]; then echo "skip $s/$ds: no samples at $out/$NPZ (run sample.sh first)"; continue; fi
        {
            echo "==== eval $s/$ds ckpt$ck on GPU$g -> $out ===="
            echo "---- L2 ASR (evaluation.py) ----"
            CUDA_VISIBLE_DEVICES="$g" python3 scripts/evaluation.py --output_dir "$out" \
                --original_dir "datasets/${ds}" --metric euclidean_l2 --model Facenet512 \
                --eval_dset test --result_file "$NPZ" --device_id "$g"
            if [ -n "${IS_PROT[$s]}" ]; then
                echo "---- Rep ASR (eval_protected.py --scheme $s) ----"
                CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval_protected.py --scheme "$s" \
                    --output_dir "$out" --result_file "$NPZ" --eval_dset test \
                    --dataset "$ds" --datasets_root datasets --device_id "$g"
            fi
            echo "==== done $s/$ds ===="
        } > "$LOGDIR/eval_${s}_${ds}.log" 2>&1 &
        i=$((i+1))
        if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
    done
done
wait
echo "evaluation done for schemes [$SCHEMES] x datasets [$DATASETS]"
