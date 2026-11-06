#!/bin/bash
# Reproduce the full attack pipeline for ONE scheme: classifier training -> GMI -> DMI.
# (These are the commands used to produce GAN_attacks/RESULTS_SUMMARY.md.)
#
# Usage: run_pipeline.sh <scheme> [gpu]
#   scheme in: unprotected | facialfe_leech | facialfe_e8 | facialfe_leech_randomised | ourfe
#   (ourfe has no differentiable surrogate -> GMI/DMI run with the identity loss not backpropping
#    to z; the attack is driven by the GAN prior only. See BiDO/model.py fallback.)
#   gpu      : CUDA device index (default 0)
#
# Note: the unprotected target/evaluator is located by globbing FaceNet_reg_facenet_*_unprotected_d.tar
# and reading the accuracy from the filename (GMI/attack.py, DMI/{k+1_gan_vib,recover_vib}.py), so no
# accuracy is hardcoded; re-training unprotected updates that checkpoint and the attacks pick it up.
set -e
SCHEME=$1
G=${2:-0}
[ -z "$SCHEME" ] && { echo "usage: $0 <scheme> [gpu]"; exit 1; }

PY="${PY:-python}"          # override with PY=/path/to/python; default = active env (conda activate ddpm)
export CUDA_VISIBLE_DEVICES=$G
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# (0) untrained starting checkpoint for FE schemes (idempotent)
cd "$ROOT/BiDO"
if [ "$SCHEME" != "unprotected" ] && [ ! -f "target_model/FaceNet_pytorch_classify_${SCHEME}_untrained.tar" ]; then
    echo "=== [0] creating untrained checkpoint for $SCHEME ==="
    CUDA_VISIBLE_DEVICES="" $PY -c "import torch,torch.nn as nn,model; \
n=model.InceptionResnetV1PipeFE(scheme='${SCHEME}',pretrained='vggface2',classify=True,num_classes=1000,device='cpu'); \
torch.save({'state_dict':nn.DataParallel(n).state_dict()},'target_model/FaceNet_pytorch_classify_${SCHEME}_untrained.tar')"
fi

# (1) classifier training -> FaceNet_reg_facenet_<acc>_<scheme>_d.tar
echo "=== [1/3] training $SCHEME classifier (GPU $G) ==="
$PY train_reg.py --dataset=celeba --defense=reg --embed_model facenet --prot "$SCHEME" --sameR d

ACC=$(ls -t "$ROOT"/BiDO/target_model/celeba/reg/facenet/FaceNet_reg_facenet_*_${SCHEME}_d.tar 2>/dev/null \
      | head -1 | sed -E "s/.*facenet_([0-9.]+)_${SCHEME}_d\\.tar/\\1/")
echo "=== $SCHEME classifier acc = $ACC ==="

# (2) GMI attack (differentiable FE for FE schemes)
echo "=== [2/3] GMI attack $SCHEME ==="
cd "$ROOT/GMI"
$PY attack.py --prot "$SCHEME" --embed_model facenet --sameR d --acc "$ACC" --verbose

# (3) DMI attack: target-specific K+1 GAN then recovery
echo "=== [3/3] DMI attack $SCHEME ==="
cd "$ROOT/DMI"
$PY k+1_gan_vib.py --dataset=celeba --defense=reg --prot "$SCHEME" --embed_model facenet --sameR d --acc "$ACC"
$PY recover_vib.py  --dataset=celeba --defense=reg --prot "$SCHEME" --embed_model facenet --sameR d --acc "$ACC" --verbose

echo "=== $SCHEME pipeline complete (acc=$ACC) ==="
