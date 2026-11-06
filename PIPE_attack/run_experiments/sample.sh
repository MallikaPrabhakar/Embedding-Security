#!/bin/bash
# Sample 1000 reconstructions from each CelebA-trained scheme model, conditioned on each
# dataset's TEST (un)protected embeddings -- in-domain (celeba) + cross-dataset (lfw,
# casia_webface). Consolidates the old sample.sh / sample_facialfe_leech.sh and the sampling
# halves of eval_crossdataset*.sh. Each scheme is sampled at its selected checkpoint.
#
#   Usage:  [SCHEMES="..."] [DATASETS="celeba lfw casia_webface"] [GPUS="0 1 2 3"] [NUM=1000] \
#           bash run_experiments/sample.sh
#   Run from PIPE_attack/ AFTER train.sh.  Outputs the reconstructed-samples npz ->
#     celeba : results/ckpt_<ckpt>/<scheme>/celeba/facenet/
#     cross  : results/crossdataset/<dataset>/ckpt_<ckpt>/<scheme>/celeba/facenet/
if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base); else echo "conda not found"; exit 1; fi
cd "$(dirname "$0")/.." || exit 1          # -> PIPE_attack/
source "$CONDA_BASE/bin/activate" ddpm

SCHEMES=${SCHEMES:-"unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe"}
DATASETS=${DATASETS:-"celeba lfw casia_webface"}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
NUM=${NUM:-1000}; BATCH=${BATCH:-100}
LOGDIR=run_experiments/logs; mkdir -p "$LOGDIR"

SAMPLE_FLAGS="--image_size 64 --attention_resolutions 32,16,8 --dropout 0.1 --class_cond True --learn_sigma True --num_channels 192 --num_heads 3 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_new_attention_order True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule cosine"
TEST_FLAGS="--batch_size $BATCH --num_samples $NUM --timestep_respacing 250"

# per-scheme: conditioning embedding size | protected scale dir | checkpoint to sample
declare -A EMB=(      [unprotected]=512     [facialfe_e8]=512                  [facialfe_leech]=504                  [facialfe_leech_randomised]=504                  [ourfe]=42 )
declare -A SCALEDIR=( [unprotected]=facenet [facialfe_e8]=facenet_invscale7.25 [facialfe_leech]=facenet_scale1461.76 [facialfe_leech_randomised]=facenet_scale1461.76 [ourfe]=facenet_scale144.64 )
declare -A CKPT=(     [unprotected]=025000  [facialfe_e8]=025000               [facialfe_leech]=050000               [facialfe_leech_randomised]=050000               [ourfe]=050000 )

res_dir() {  # scheme dataset ckpt -> result/sample dir
    if [ "$2" = celeba ]; then echo "results/ckpt_$3/$1/celeba/facenet"
    else echo "results/crossdataset/$2/ckpt_$3/$1/celeba/facenet"; fi
}

i=0
for s in $SCHEMES; do
    if [ -z "${EMB[$s]}" ]; then echo "skip $s: unknown scheme"; continue; fi
    for ds in $DATASETS; do
        g=${GPUS[$((i % ${#GPUS[@]}))]}; ck=${CKPT[$s]}
        out=$(res_dir "$s" "$ds" "$ck"); mkdir -p "$out"
        echo "sample $s/$ds ckpt$ck on GPU$g -> $out"
        python3 scripts/image_sample.py $SAMPLE_FLAGS $TEST_FLAGS --eval_dset test \
            --model_path "models/${s}/celeba/facenet/model${ck}.pt" \
            --checkpoint_dir "$out" --embedding_size "${EMB[$s]}" \
            --embeddings_path "datasets/${s}/${ds}/${SCALEDIR[$s]}/test/embeddings" \
            --device_id "$g" > "$LOGDIR/sample_${s}_${ds}.log" 2>&1 &
        i=$((i+1))
        if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
    done
done
wait
echo "sampling done for schemes [$SCHEMES] x datasets [$DATASETS]"
