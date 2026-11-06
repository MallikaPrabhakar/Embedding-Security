#!/bin/bash
# Train the PIPE class-conditional diffusion inversion attacker on CelebA, for every scheme.
# Consolidates the old train_unprotected.sh / train_facialfe.sh / train_ourfe.sh (+ the two
# leech schemes). Each scheme conditions on its own celeba TRAIN (un)protected embeddings.
# One GPU per scheme, cycled over $GPUS; runs in the background with a per-scheme log and
# barriers when there are fewer GPUs than schemes.
#
#   Usage:  [SCHEMES="..."] [GPUS="0 1 2 3"] bash run_experiments/train.sh
#   Run from PIPE_attack/.  Outputs -> models/<scheme>/celeba/facenet/model<step>.pt
#
# NOTE: no `set -u` -- conda's activate.d hooks reference unbound vars (e.g. SYS_SYSROOT) that
# abort `source activate` under set -u.
if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base); else echo "conda not found"; exit 1; fi
cd "$(dirname "$0")/.." || exit 1          # -> PIPE_attack/
source "$CONDA_BASE/bin/activate" ddpm

SCHEMES=${SCHEMES:-"unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe"}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
LOGDIR=run_experiments/logs; mkdir -p "$LOGDIR"

MODEL_FLAGS="--image_size 64 --num_channels 192 --num_heads 3 --num_res_blocks 3 --attention_resolutions 32,16,8 --dropout 0.1 --class_cond True --learn_sigma True --resblock_updown True --use_fp16 True --use_new_attention_order True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"

# per-scheme: conditioning embedding size | celeba train-emb scale dir | lr_anneal_steps | save_interval
declare -A EMB=(      [unprotected]=512               [facialfe_e8]=512                  [facialfe_leech]=504                  [facialfe_leech_randomised]=504                  [ourfe]=42 )
declare -A SCALEDIR=( [unprotected]=facenet           [facialfe_e8]=facenet_invscale7.25 [facialfe_leech]=facenet_scale1461.76 [facialfe_leech_randomised]=facenet_scale1461.76 [ourfe]=facenet_scale144.64 )
declare -A ANNEAL=(   [unprotected]=025000            [facialfe_e8]=025000               [facialfe_leech]=050000               [facialfe_leech_randomised]=050000               [ourfe]=050000 )
declare -A SAVE=(     [unprotected]=025000            [facialfe_e8]=025000               [facialfe_leech]=025000               [facialfe_leech_randomised]=025000               [ourfe]=002500 )

i=0
for s in $SCHEMES; do
    if [ -z "${EMB[$s]}" ]; then echo "skip $s: unknown scheme"; continue; fi
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    ckpt_dir=models/${s}/celeba/facenet; mkdir -p "$ckpt_dir"
    TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --ema_rate 0.9999 --lr_anneal_steps ${ANNEAL[$s]} --save_interval ${SAVE[$s]}"
    echo "launch train $s on GPU$g  emb=${EMB[$s]} steps=${ANNEAL[$s]} save=${SAVE[$s]}"
    python3 scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
        --data_dir datasets/celeba/train \
        --emb_dir "datasets/${s}/celeba/${SCALEDIR[$s]}/train" \
        --checkpoint_dir "$ckpt_dir" --embedding_size "${EMB[$s]}" --device_id "$g" \
        > "$LOGDIR/train_${s}.log" 2>&1 &
    i=$((i+1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "training done for: $SCHEMES"
