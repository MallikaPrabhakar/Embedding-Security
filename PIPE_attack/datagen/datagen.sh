if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

source $CONDA_BASE/bin/activate ddpm

# split and obtain appropriate subset from root directory datasets/ folder for images and unprotected embeddings
python datasplit.py --dataset celeba
python datasplit.py --dataset lfw
python datasplit.py --dataset casia_webface

# generate protected embeddings for each protection scheme
python datagen_facialfe_e8.py --dataset celeba --model facenet &
python datagen_facialfe_e8.py --dataset lfw --model facenet &
python datagen_facialfe_e8.py --dataset casia_webface --model facenet
python datagen_facialfe_leech.py --dataset celeba --model facenet &
python datagen_facialfe_leech.py --dataset casia_webface --model facenet  &
python datagen_facialfe_leech.py --dataset lfw --model facenet 
python datagen_facialfe_leech_randomised.py --dataset celeba --model facenet &
python datagen_facialfe_leech_randomised.py --dataset lfw --model facenet  &
python datagen_facialfe_leech_randomised.py --dataset casia_webface --model facenet 
python datagen_ourdef.py --dataset celeba --model facenet &
python datagen_ourdef.py --dataset lfw --model facenet  &
python datagen_ourdef.py --dataset casia_webface --model facenet

source $CONDA_BASE/bin/activate sage

python datagen_randomfe.py --dataset celeba --model facenet &
python datagen_randomfe.py --dataset lfw --model facenet  &
python datagen_randomfe.py --dataset casia_webface --model facenet 