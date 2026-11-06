# preprocess images from raw datasets
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

source $CONDA_BASE/bin/activate ddpm

python celeba_pp_filtered.py
python lfw_pp_filtered.py
python casia_webface_pp_filtered.py


# generate embeddings for processed images
source $CONDA_BASE/bin/activate deepface

python generate_embeddings.py --dataset celeba --model Facenet512
python generate_embeddings.py --dataset lfw --model Facenet512
python generate_embeddings.py --dataset casia_webface --model Facenet512