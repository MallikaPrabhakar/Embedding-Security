if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

# Evaluate Bob's reconstructed images for all 5 facenet PIPE-FE schemes (run after reconstruct.sh).
ORIG_DIR="DataGen/databases/celeba/facenet"

TAGS=(unprotected facialfe_e8 facialfe_leech_centered facialfe_leech_randomised_centered ourfe)
GPUS=(0 1 2 3)

# ---- evaluate reconstructions (deepface env, Facenet512 euclidean_l2) ----
source $CONDA_BASE/bin/activate deepface
EVAL_FACENET="--samples 1000 --model Facenet512 --metric euclidean_l2 --original_dir $ORIG_DIR --output_dir results/facenet"
i=0
for t in "${TAGS[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  python evaluation.py ${EVAL_FACENET} --same_r none --prot_scheme "$t" --eval_dset test --device_id "$g" &
  i=$((i+1))
done
wait
echo "evaluation done for: ${TAGS[*]}"
