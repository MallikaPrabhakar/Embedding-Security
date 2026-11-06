# GAN Attacks
<!-- label - message - color -->

This directory contains existing model inversion (MI) attacks such as the GMI attack and the KED-MI attack. The code in this directory is a modified subset of the [code](https://github.com/AlanPeng0897/Defend_MI) by the authors of the BiDO paper: *[Bilateral Dependency Optimization: Defending Against Model-inversion Attacks](https://arxiv.org/pdf/2206.05483.pdf)*. We would like to thank the authors for making their code publicly available. 

```bibtex
@inproceedings{peng2022BiDO,
title={Bilateral Dependency Optimization: Defending Against Model-inversion Attacks},
author={Peng, Xiong and Liu, Feng and Zhang, Jingfeng and Lan, Long and Ye, Junjie and Liu, Tongliang and Han, Bo},
booktitle={KDD},
year={2022}
}
```

# Results

Attack results are summarised in [RESULTS_SUMMARY.md](./RESULTS_SUMMARY.md) — classifier accuracy plus GMI / KED-MI embedding ASR for each scheme. To regenerate that table from the saved reconstructions, run `scripts/aggregate_results.sh. The **Reproducing RESULTS_SUMMARY.md** section below documents the full train → attack → aggregate results pipeline.

# Setup


### Environment

We use the `ddpm` environment. Instructions to set it up is available in the [main README.md](../README.md). 

### Dataset

We use a subset of the preprocessed CelebA images used by PIPE-Attack. Please follow instructions in the [main README.md](../README.md) to download the preprocessed CelebA dataset in `Embedding-Security/datasets/pp/celeba`. 

### Schemes

Each protection scheme is a fuzzy-extractor (FE) transform applied to the 512-d FaceNet embedding before the classifier head (ported from `Bob_attack/experiments/pipe_fe_transforms.py`). The five schemes reported in `RESULTS_SUMMARY.md`:

| scheme | helper dim | seeded | notes |
|---|---|---|---|
| `unprotected` | 512 | no | baseline; the raw embedding |
| `facialfe_e8` | 512 | no | E8-lattice sketch; ~no protection |
| `facialfe_leech` | 504 | no | Leech secure sketch |
| `facialfe_leech_randomised` | 504 | yes | random per-image codeword |
| `ourfe` | 42 | yes | our defense; decode-first keyed-hash sketch |

"Seeded" schemes derive their codeword/keys from the image filename (per-image randomness).

### Target Models

Both GMI and KED-MI attack experiments require an **evaluation model** and a **target classifier model**. Each **protection scheme** corresponds to a distinct target model, with the appropriate protection layer appended before the final classification layer.


Run the [initialize models notebook](./BiDO/initialize_model.ipynb) to construct the evaluation model and/or untrained target classifier models. Then, train a target classifier with:
```
# assuming you are in the GAN_attacks directory
cd BiDO
conda activate ddpm
python train_reg.py --dataset=celeba --defense=reg --embed_model facenet --prot {prot_scheme} --sameR d
```
`{prot_scheme}` is one of the [schemes](#schemes) above. The FaceNet backbone is frozen (`vggface2` weights); only an auto-sized linear head is trained on the protected helper. The checkpoint lands at `target_model/celeba/reg/facenet/FaceNet_reg_facenet_<acc>_<prot>_d.tar` (`<acc>` = best test accuracy). `--sameR` is fixed to `d` (independent randomness per enrolment) throughout.

> The seeded schemes derive their codeword/keys from the image filename, i.e. per-image randomness. Because the randomness is per image, it carries no identity signal that generalises across an identity's images, so `facialfe_leech_randomised` trains to ~chance accuracy. This is expected and is exactly what `RESULTS_SUMMARY.md` reports.


### Attack Models

For GMI attack, we use the pretrained GAN model provided by [BiDO paper's authors](https://github.com/AlanPeng0897/Defend_MI). Download the [pretrained GAN model](https://drive.google.com/drive/folders/1uMvltKbFOuAgK4wtxusd2ADwHq3prKxz?usp=sharing) into `GMI/GAN_models/celeba/`. 


For the KED-MI (DMI) attack, a GAN is trained specific to each target classifier. These are produced automatically by the pipeline; to train one manually:
```
# assuming you are in the GAN_attacks directory
cd DMI
conda activate ddpm
python k+1_gan_vib.py --dataset=celeba --defense=reg --embed_model facenet --prot {prot_scheme} --sameR d --acc {acc}
```
where `{acc}` is the trained classifier's accuracy tag (the `<acc>` in its checkpoint filename). The GAN is written under `DMI/improvedGAN/celeba/reg/facenet/`.

# Reproducing RESULTS_SUMMARY.md

First complete [Setup](#setup) (dataset, evaluation model, GMI GAN). Then, from `GAN_attacks/` with the `ddpm` env active:

```
conda activate ddpm
bash scripts/run_all.sh          # all 5 schemes (train -> GMI -> DMI), then aggregate
```

`run_all.sh` launches one `run_pipeline.sh <scheme> <gpu>` per scheme in parallel (edit the GPU indices to match your machine) and finally calls `aggregate_results.sh`, which writes [`RESULTS_SUMMARY.md`](./RESULTS_SUMMARY.md). To run a single scheme on one GPU:

```
bash scripts/run_pipeline.sh facialfe_leech 0      # train classifier -> GMI -> DMI for `facialfe_leech`
```

Each `run_pipeline.sh` does: **(0)** build the untrained checkpoint, **(1)** `train_reg.py` (train target classifier), **(2)** `GMI/attack.py` (launch GMI attack), **(3)** `DMI/k+1_gan_vib.py` + `DMI/recover_vib.py` (train GAN model + launch KED-MI attack). Reconstructed images are written to `{GMI|DMI}/attack_res/celeba/reg/facenet/<scheme>_d/all/` as `attack_iden_<id>|<seed>.png` (1000 identities x 5 seeds = 5000 each); the optimized latents are saved alongside in `.../z/`.

### Metrics

`aggregate_results.sh` runs `gather_results.py`, which scores the saved reconstructions and writes the table. For every scheme it reports, per reconstructed image:

- **classifier acc** — the scheme's trained classifier test accuracy (from its checkpoint name)
- **ASR-L2** (GMI / DMI) — a reconstruction succeeds if the FaceNet *evaluation* embedding's `euclidean_l2` (L2 of unit-normalised embeddings) to the target identity's train-time embedding is `< 1.1027` (the Facenet512 threshold).

### Caveats (read before interpreting the table)

- **`unprotected` ignores `--acc`.** For the `unprotected` scheme both the evaluator and the target are the trained unprotected classifier, which `attack.py` locates automatically (globbing `FaceNet_reg_facenet_*_unprotected_d.tar` and reading the accuracy from the filename); the `--acc` tag is only consulted for the FE schemes.

Individual attacks can also be launched directly:
```
# GMI (from GAN_attacks/)
cd GMI && python attack.py --prot {prot_scheme} --embed_model facenet --sameR d --acc {acc} --verbose
# KED-MI / DMI recovery (after its k+1 GAN is trained)
cd DMI && python recover_vib.py --dataset=celeba --defense=reg --prot {prot_scheme} --embed_model facenet --sameR d --acc {acc} --verbose
```