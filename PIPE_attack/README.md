# PIPE-Attack

This repository contains code for the PIPE attack used for inversion of protected embeddings. This repository is forked from [openai/guided-diffusion](https://github.com/openai/guided-diffusion), modified to use embeddings instead of class as the condition vector.

## Results
To check all ASR values, run the jupyter notebook [ASR.ipynb](ASR.ipynb).

Precomputed random-guessing (L2 + Rep) and TPR/FPR baseline values for every scheme are in [random_guessing_results.csv](random_guessing_results.csv) and [tpr_fpr_results.csv](tpr_fpr_results.csv) (see [Random-guessing baseline & TPR/FPR](#random-guessing-baseline--tprfpr)).

Recovered-secret entropy for each scheme is in [datagen/entropy_results/RESULTS.md](datagen/entropy_results/RESULTS.md).

## Datasets
After obtaining preprocessed datasets `Embedding-Security/datasets/pp`, run the following commands to obtain all protected embeddings:
```bash
# assuming you are in PIPE_attack directory
cd datagen

# to generate embeddings for all settings used in our paper
bash datagen.sh

# to generate protected embeddings for a specific setting
conda activate deepface

python datasplit.py --dataset {dataset}

# one of the per-scheme generators:
python datagen_facialfe_e8.py               --dataset {dataset} --model facenet   # facialfe (E8 secure sketch)
python datagen_facialfe_leech.py            --dataset {dataset} --model facenet   # facialfe (Leech secure sketch)
python datagen_facialfe_leech_randomised.py --dataset {dataset} --model facenet   # facialfe (randomised Leech)
python datagen_ourdef.py                    --dataset {dataset} --model facenet   # ourfe (our defense)
python datagen_randomfe.py                  --dataset {dataset} --model facenet   # randomfe (FOWH/LWE, sage env; numpy alt: datagen_randomfe_numpy.py)
```
Where `{dataset}` can be `celeba`, `lfw` or `casia_webface`. All embeddings use the `facenet` model.

> **randomfe is not a PIPE reconstruction target.** No PIPE model is trained or sampled on it; it is
> used only for the TPR/FPR (`tpr_fpr.py`) and random-guessing (`random_guessing.py`) baselines.
> randomfe datagen runs in the `sage` environment (`datagen_randomfe.py`); the pure-numpy alternative
> (`datagen_randomfe_numpy.py`, no `sage`) implements the **same scheme**, but as numpy's RNG differs
> from sage's `GF(q)` RNG, its `A` and per-identity secret `b` form a *different* random instance.
> randomfe reads its FOWH parameters from `datagen/randomfe_parameters.json` and emits only
> `p/` + `codewords/` (plus the shared `A` / `projection_matrix` / `lll_basis`) — no conditioning embeddings.

> **Leech decoder (`facialfe_leech*`).** These schemes decode with the [`leech-decoding`](https://github.com/avanpo/leech-decoding) git submodule (Vardy–Be'ery maximum-likelihood decoder). Initialize it once from the repo root with `git submodule update --init leech-decoding`. On first import, `datagen/leech_helper.py` compiles its C library automatically — `gcc -O2 -fPIC -shared leech.c leech_utils.c -o libleech.so` — so a C compiler (`gcc`) must be on `PATH`; or build it manually with that command from inside `leech-decoding/`.

## Training
For training a PIPE reconstruction model for a `{prot_scheme}`, run the following commands:
```bash
# assuming you are in PIPE_attack directory

# to train all models (every scheme) used in our paper
bash run_experiments/train.sh

# to train model for a specific setting
conda activate ddpm

MODEL_FLAGS="--image_size 64 --num_channels 192 --num_heads 3 --num_res_blocks 3 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --resblock_updown True --use_fp16 True --use_new_attention_order True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --ema_rate 0.9999 --lr_anneal_steps 025000 --save_interval 025000"

python3 scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --data_dir datasets/celeba/train --emb_dir datasets/{prot_scheme}/celeba/{embedding_dir}/train --checkpoint_dir models/{prot_scheme}/celeba/facenet --embedding_size {length} --device_id 0
```

Here all variables are the same as before. The value of flag `--embedding_size` depends on `{prot_scheme}`: it is `512` for `facialfe_e8` and `unprotected`, `504` for the Leech-based `facialfe` variants, and `42` for `ourfe`. See `run_experiments/train.sh` (the `EMB`/`SCALEDIR` maps) for the per-scheme values.



## Sampling and Evaluation
After obtaining the trained models, to reconstruct images from embeddings, run the following commands


```bash
# assuming you are in PIPE_attack directory

# to reconstruct all images used in our paper
bash run_experiments/sample.sh

# to evaluate (pairwise distances and plots) all images used in our paper
bash run_experiments/eval.sh

# to sample images for a specific setting
conda activate ddpm

SAMPLE_FLAGS="--image_size 64 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --num_channels 192 --num_heads 3 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_new_attention_order True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule cosine"
TEST_FLAGS="--batch_size 100 --num_samples 1000 --timestep_respacing 250"

python3 scripts/image_sample.py $SAMPLE_FLAGS $TEST_FLAGS --model_path models/{prot_scheme}/celeba/facenet/model025000.pt --checkpoint_dir results/{prot_scheme}/{dataset}/facenet --embedding_size {length} --embeddings_path datasets/{prot_scheme}/{dataset}/facenet/test/embeddings --eval_dset test --device_id 0

# to compute pair_wise distances and plots for a specific setting
conda activate deepface

EVAL_FLAGS="--metric euclidean_l2 --model Facenet512"
python3 scripts/evaluation.py --output_dir results/{prot_scheme}/{dataset}/facenet --original_dir datasets/{dataset} $EVAL_FLAGS --device_id 2 --eval_dset test
```

Here, `dataset` can be `celeba`, `lfw` or `casia_webface` as needed for evaluation.

To measure recovery of the protected codeword (the FE secret) rather than face-matching, use `scripts/eval_protected.py` (schemes: `facialfe_e8`, `facialfe_leech`, `facialfe_leech_randomised`, `ourfe`).

## Random-guessing baseline & TPR/FPR

`random_guessing.py` and `tpr_fpr.py` score the chance-level (no trained PIPE model) L2 and
Rep metrics for every scheme. Both are pure numpy (no `sage`). `random_guessing.py` is the
leave-one-out baseline (each of the 1000 test embeddings authenticates as every other →
999,000 attempts); `tpr_fpr.py` reports TPR over genuine pairs and FPR over impostor pairs.
Run the sweeps from `PIPE_attack/`:

```bash
# random-guessing baseline, TEST split, all schemes x all datasets -> random_guessing_results.csv
bash run_random_guessing.sh

# TPR / FPR (1000 genuine + 1000 impostor pairs each) -> tpr_fpr_results.csv
bash run_tpr_fpr.sh
```

Both scripts cover all six schemes (`unprotected`, `facialfe_e8`, `facialfe_leech`,
`facialfe_leech_randomised`, `ourfe`, `randomfe`) and accept `--scheme`/`--dataset` for a
single run; see each file's header for options.
