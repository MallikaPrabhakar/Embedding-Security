# Bob attack
We have adapted the repository from [Bob](https://www.idiap.ch/software/bob) by the authors of "Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack".

Cite the original version for use.
```BibTeX
@article{tifs2024vulnerability,
  title={Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack},
  author={Shahreza, Hatef Otroshi and Hahn, Vedrana Krivoku{\'c}a and Marcel, S{\'e}bastien},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```

## Results
To check all ASR values, run the jupyter notebook [experiments/ASR.ipynb](experiments/ASR.ipynb).

## Setup
To build the Bob environment, run the following:
```bash
conda create --name bob --file package-list.txt
conda activate bob
buildout
```

Please note that Bob uses the same set of images from CelebA as PIPE_attack which is a smaller subset of preprocessed images, please run 
```bash
# assuming you are in the Bob_attack directory
cd ../PIPE_attack/datagen/
python datasplit.py
cd ../../Bob_attack/
```

## Datagen
After generating or downloading the preprocessed dataset in `Embedding-Security/PIPE_attack/datasets/`, run the following commands to generate dataset and embeddings

```bash
# assuming you are in the Bob_attack directory
cd experiments/DataGen

# to generate embeddings for all settings used in our paper (facenet)
bash datagen.sh

# to generate protected embeddings for a specific setting
conda activate bob
python GenDataset.py --same_r none --prot_scheme {prot_scheme} --mode {mode}    # add --centered for the leech sketches
```

Here `{mode}` can be `train, test, test_seen`. All embeddings use the `facenet` model. `{prot_scheme}` can be `unprotected, facialfe_e8, facialfe_leech, facialfe_leech_randomised` or `ourfe` (our defense). The leech sketches (`facialfe_leech`, `facialfe_leech_randomised`) are generated with `--centered` (the mod-q centered public helper).

## Training

For training a Bob reconstruction model for a {prot_scheme}, run the following commands:
```bash
# assuming you are in the Bob_attack directory
cd experiments/TrainNetwork

# to train all models used in our paper
bash train.sh            

# to train model for a specific setting
conda activate bob

python train.py --model facenet --prot_scheme {prot_scheme} --same_r none --data_dir ../DataGen/databases/celeba/facenet/{prot_scheme} --length_of_embedding {length} --device cuda:0
```
Here all variables are the same as before. The length of embedding depends on `{prot_scheme}`: `512` for `unprotected` / `facialfe_e8`, `504` for the leech sketches, and `42` for `ourfe`. See `train.sh` for more details.

## Sampling and evaluation

After obtaining the trained models, to reconstruct images from embeddings, run the following commands


```bash
# assuming you are in the Bob_attack directory
cd experiments

# to reconstruct images for all settings used in our paper
bash reconstruct.sh

# to compute pair_wise distances and plots for all settings (run after reconstruct.sh)
bash eval.sh

# to reconstruct images for a specific setting
conda activate bob

COMMON_FACENET="--generator_checkpoint 200 --original_dir DataGen/databases/celeba/facenet --model facenet"

python reconstruct_images.py ${COMMON_FACENET} --same_r none --prot_scheme {prot_scheme} --embedding_size {length} --eval_dset test


# to compute pair_wise distances and plots for a specific setting
conda activate deepface

EVAL_FACENET="--samples 1000 --model Facenet512 --metric euclidean_l2 --original_dir DataGen/databases/celeba/facenet --output_dir results/facenet"

python evaluation.py ${EVAL_FACENET} --same_r none --prot_scheme {prot_scheme}  --eval_dset test --device_id 0
```

For `reconstruct_images.py` / `evaluation.py`, `{prot_scheme}` is the output-dir tag: pass the `_centered` tag for the leech sketches (`facialfe_leech_centered`, `facialfe_leech_randomised_centered`), and `{length}` is the embedding length above (`512` / `504` / `42`). `reconstruct.sh` and `eval.sh` set both automatically for every scheme.
