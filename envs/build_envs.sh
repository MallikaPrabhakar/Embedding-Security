#!/usr/bin/env bash
# Build all conda environments used by the artifact. Run from the repository root: `bash envs/build_envs.sh`

conda env create -f ./envs/deepface.yml
conda env create -f ./envs/ddpm.yml
conda env create -f ./envs/sage.yml

# --- ddpm ---
source "$(conda info --base)/etc/profile.d/conda.sh"
cd PIPE_attack
conda activate ddpm
pip install -e .
conda deactivate
cd ..

# --- bob ---
source "$(conda info --base)/etc/profile.d/conda.sh"
cd Bob_attack
conda create --yes --name bob --file package-list.txt
conda activate bob
buildout
conda deactivate
cd ..
