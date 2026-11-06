#!/bin/bash
# Train Bob inversion models on the PIPE-FE datasets in DataGen/databases/celeba/facenet.
# Usage: train.sh [scheme ...] [-- gpu ...]
#   default schemes: unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe
#   default gpus:    0 1 2 3   (one GPU per scheme, all in parallel; cycled + barriered
#                               every ${#gpus[@]} launches when there are fewer GPUs than schemes)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bob
cd "$(dirname "$0")" || exit 1          # run from TrainNetwork/ so `python train.py` resolves
LOG="../log-files"
mkdir -p "$LOG"
# per-scheme flattened embedding length (--length_of_embedding)
declare -A LEN=( [unprotected]=512 [facialfe_e8]=512 [facialfe_leech]=504 [facialfe_leech_randomised]=504 [ourfe]=42 )
# schemes whose datagen emitted the centered mod-q helper (dir tagged _centered, train --centered)
declare -A CENTERED=( [facialfe_leech]=1 [facialfe_leech_randomised]=1 )

schemes=(); gpus=(); seen_dd=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_dd=1; continue; fi
  if [ $seen_dd -eq 0 ]; then schemes+=("$a"); else gpus+=("$a"); fi
done
[ ${#schemes[@]} -eq 0 ] && schemes=(unprotected facialfe_e8 facialfe_leech facialfe_leech_randomised ourfe)
[ ${#gpus[@]} -eq 0 ] && gpus=(0 1 2 3)

i=0
for s in "${schemes[@]}"; do
  if [ -z "${LEN[$s]}" ]; then
    echo "skip $s: unknown scheme (known: ${!LEN[*]})"; continue
  fi
  if [ -n "${CENTERED[$s]}" ]; then
    tag="${s}_centered_none"; cflag="--centered"
  else
    tag="${s}_none"; cflag=""
  fi
  data_dir="../DataGen/databases/celeba/facenet/${tag}"
  if [ ! -d "$data_dir" ]; then
    echo "skip $s: data dir not found ($data_dir)"; continue
  fi
  g=${gpus[$((i % ${#gpus[@]}))]}
  echo "launch train $s on cuda:$g len=${LEN[$s]} dir=$tag"
  python train.py --model facenet --prot_scheme "$s" --same_r none $cflag \
      --data_dir "$data_dir" \
      --length_of_embedding "${LEN[$s]}" --device "cuda:$g" \
      > "$LOG/train_${s}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % ${#gpus[@]})) -eq 0 ]; then wait; fi
done
wait
echo "TRAINING DONE for: ${schemes[*]}"
