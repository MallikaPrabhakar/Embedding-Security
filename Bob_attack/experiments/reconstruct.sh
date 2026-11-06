if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

source $CONDA_BASE/bin/activate bob

# Reconstruct Bob's inversion-attack images for all 5 facenet PIPE-FE schemes (run eval.sh after).
ORIG_DIR="DataGen/databases/celeba/facenet"
GEN_CKPT=200
MODEL=facenet
TAGS=(unprotected facialfe_e8 facialfe_leech_centered facialfe_leech_randomised_centered ourfe)
declare -A DIM=(
  [unprotected]=512
  [facialfe_e8]=512
  [facialfe_leech_centered]=504
  [facialfe_leech_randomised_centered]=504
  [ourfe]=42
)

# ---- reconstruct images (bob env) ----
for t in "${TAGS[@]}"; do
  python reconstruct_images.py --generator_checkpoint $GEN_CKPT --original_dir "$ORIG_DIR" --model $MODEL \
      --same_r none --prot_scheme "$t" --eval_dset test --embedding_size "${DIM[$t]}" &
done
wait
echo "reconstruction done for: ${TAGS[*]}"
