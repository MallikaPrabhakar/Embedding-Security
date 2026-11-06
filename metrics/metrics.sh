if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

source $CONDA_BASE/bin/activate ddpm
cd "$(dirname "$0")" || exit 1          # run from metrics/ (relative fid/ + csv paths)

echo "Calculating FID Scores for PIPE and Bob attack..."

python FID.py

# clean display names (match lpips_results.csv) paired index-wise with each attack's recon dirs
disp=("unprotected" "facialfe_e8" "facialfe_leech" "facialfe_leech_randomised" "ourfe")
bob_prot_schemes=("unprotected_none" "facialfe_e8_none" "facialfe_leech_centered_none" "facialfe_leech_randomised_centered_none" "ourfe_none")
pipe_prot_schemes=("unprotected" "facialfe_e8" "facialfe_leech" "facialfe_leech_randomised" "ourfe")
gan_schemes=("unprotected" "facialfe_e8" "facialfe_leech" "facialfe_leech_randomised" "ourfe")
models=("facenet")

FID_CSV=${FID_CSV:-fid_results.csv}
echo "attack,model,scheme,fid" > "$FID_CSV"

# run pytorch_fid on an (original, reconstructed) folder pair, parse the "FID:  <v>" it prints,
# echo it and append a row to $FID_CSV. Skips (records NA) when a folder is missing.
fid_to_csv() {  # attack model scheme orig recon
  local attack=$1 model=$2 scheme=$3 orig=$4 recon=$5
  if [ ! -d "$orig" ] || [ ! -d "$recon" ]; then
    echo "FID for $attack $model $scheme: SKIP (missing $orig or $recon)"
    echo "$attack,$model,$scheme,NA" >> "$FID_CSV"
    return
  fi
  local out fid
  out=$(python -m pytorch_fid "$orig" "$recon" 2>/dev/null)
  fid=$(printf '%s\n' "$out" | sed -nE 's/.*FID:[[:space:]]*([0-9.]+).*/\1/p' | tail -1)
  [ -z "$fid" ] && fid=$(printf '%s\n' "$out" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "FID for $attack $model $scheme: ${fid:-NA}"
  echo "$attack,$model,$scheme,${fid:-NA}" >> "$FID_CSV"
}

for model in "${models[@]}"; do
  echo "Calculating FID Scores for Bob $model reconstructions..."
  for i in "${!disp[@]}"; do
    fid_to_csv "Bob" "$model" "${disp[$i]}" "fid/original_bob/" "fid/reconstructed_bob/$model/${bob_prot_schemes[$i]}"
  done
  echo "Calculating FID Scores for PIPE $model reconstructions..."
  for i in "${!disp[@]}"; do
    fid_to_csv "PIPE" "$model" "${disp[$i]}" "fid/original_pipe/" "fid/reconstructed_pipe/$model/${pipe_prot_schemes[$i]}"
  done
done

# GMI / KED-MI (facenet only); originals shared in fid/original_gan/
for src in gmi kedmi; do
  echo "Calculating FID Scores for $src reconstructions..."
  for scheme in "${gan_schemes[@]}"; do
    fid_to_csv "${src^^}" "facenet" "$scheme" "fid/original_gan/" "fid/reconstructed_$src/facenet/$scheme"
  done
done

echo "wrote $FID_CSV"
cat "$FID_CSV"


echo "Calculating LPIPS Scores for PIPE, Bob, GMI and KED-MI attacks..."

python LPIPS.py
