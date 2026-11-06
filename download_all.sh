#!/usr/bin/env bash
# Download the preprocessed images, protected embeddings, trained models and computed results
# from four sources and place each piece where the attack/metric code reads it:
#   1. pp datasets ............ Kaggle  snapperfe-preprocessed-dataset
#   2. PIPE attack ............ Kaggle  snapperfe-pipe-attack       (datasets + models + results)
#   3. prior work ............. Kaggle  snapperfe-prior-work        (Bob datasets/models/results + GAN models)
#   4. GAN attack results ..... Google Drive                       (reconstruction images)
#
# Schemes covered: unprotected / facialfe_e8 / facialfe_leech / facialfe_leech_randomised / ourfe
# (the PIPE attack additionally contains randomfe for tpr fpr only). All embeddings use Facenet512.
#
# Run from the repository root. These are several large files; you may comment out a section to
# download only a subset, depending on your disk space and bandwidth.

set -e

# 1. pp datasets (preprocessed images + protected embeddings)
curl -L -o pp.zip https://www.kaggle.com/api/v1/datasets/download/pipeattack/snapperfe-preprocessed-dataset
unzip -q pp.zip -d ./_dl_pp && rm pp.zip
mkdir -p ./datasets/pp
# cp -a (not mv) so the download merges into datasets/pp/celeba, which already ships
cp -a ./_dl_pp/pp/. ./datasets/pp/ && rm -r ./_dl_pp

# 2. PIPE attack: datasets + models + results
curl -L -o pipe.zip https://www.kaggle.com/api/v1/datasets/download/pipeattack/snapperfe-pipe-attack
unzip -q pipe.zip -d ./_dl_pipe && rm pipe.zip
mkdir -p ./PIPE_attack/datasets ./PIPE_attack/models ./PIPE_attack/results
mv ./_dl_pipe/datasets_pipe/datasets/* ./PIPE_attack/datasets/
mv ./_dl_pipe/models_pipe/models/*     ./PIPE_attack/models/
mv ./_dl_pipe/results_pipe/results/*   ./PIPE_attack/results/
rm -r ./_dl_pipe

# 3. Prior work: Bob attack datasets/models/results + GAN attack models
curl -L -o prior.zip https://www.kaggle.com/api/v1/datasets/download/pipeattack/snapperfe-prior-work
unzip -q prior.zip -d ./_dl_prior && rm prior.zip
mkdir -p ./Bob_attack/experiments/DataGen/databases \
         ./Bob_attack/experiments/TrainNetwork/training_files \
         ./Bob_attack/experiments/results \
         ./GAN_attacks/BiDO/target_model/celeba/reg ./GAN_attacks/BiDO/eval_model \
         ./GAN_attacks/GMI/GAN_models/celeba ./GAN_attacks/DMI/improvedGAN/celeba/reg
# Bob attack datasets, models, results
mv ./_dl_prior/Bob_attack/experiments/DataGen/databases_bob/databases/*           ./Bob_attack/experiments/DataGen/databases/
mv ./_dl_prior/Bob_attack/experiments/TrainNetwork/models_bob/training_files/*    ./Bob_attack/experiments/TrainNetwork/training_files/
mv ./_dl_prior/Bob_attack/experiments/results_bob/results/*                       ./Bob_attack/experiments/results/
# GAN attack models (target models, BiDO eval model, GMI and KED-MI GAN models)
mv ./_dl_prior/GAN_attacks/gan_attacks_models/BiDO/target_model/celeba/reg/*      ./GAN_attacks/BiDO/target_model/celeba/reg/
mv ./_dl_prior/GAN_attacks/gan_attacks_models/BiDO/eval_model/*                   ./GAN_attacks/BiDO/eval_model/
mv ./_dl_prior/GAN_attacks/gan_attacks_models/GMI/GAN_models/celeba/*             ./GAN_attacks/GMI/GAN_models/celeba/
# KED-MI (DMI): archive nests an extra celeba (improvedGAN/celeba/celeba/reg); collapse to celeba/reg
mv ./_dl_prior/GAN_attacks/gan_attacks_models/DMI/improvedGAN/celeba/celeba/reg/* ./GAN_attacks/DMI/improvedGAN/celeba/reg/
rm -r ./_dl_prior

# 4. GAN attack results (GMI/KED-MI reconstructed images, read by GAN_attacks/gather_results.py and metrics/)
curl -L -o gan_results.zip "https://drive.usercontent.google.com/download?id=1dVdhpV72v2MkQjbRc50-hoH2llmO4mnt&export=download&confirm=t"
unzip -q gan_results.zip -d ./_dl_gan && rm gan_results.zip
mkdir -p ./GAN_attacks/DMI/attack_res ./GAN_attacks/GMI/attack_res
mv ./_dl_gan/DMI/attack_res/* ./GAN_attacks/DMI/attack_res/
mv ./_dl_gan/GMI/attack_res/* ./GAN_attacks/GMI/attack_res/
rm -r ./_dl_gan
