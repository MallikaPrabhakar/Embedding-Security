# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import argparse
parser = argparse.ArgumentParser(description='Train face reconstruction network')
parser.add_argument('--data_dir', metavar='<data_dir>', type= str, default='lfw_pp',
                    help='dataset to be used')
parser.add_argument('--prot_scheme', metavar='<prot_scheme>', type= str, default='unprotected',
                    help='protection scheme to be used')
parser.add_argument('--same_r', metavar='<same_r>', type= str, default='none',
                    help='takes values, none, true, false')
parser.add_argument('--length_of_embedding', metavar='<length_of_embedding>', type= int, default=512,
                    help='length of embedding')
parser.add_argument('--device', metavar='<device>', type= str, default='cuda',
                    help='device to be used, cuda or cpu')
parser.add_argument('--model', metavar='<model>', type= str, default='facenet',
                    help='face recognition model to be used (facenet)')
parser.add_argument('--centered', action='store_true',
                    help='centered-sketch run: only tags the output dir as <scheme>_centered_<same_r> '
                         '(prot_scheme stays the base name so the ID loss picks the right recognizer)')
args = parser.parse_args()

import os,sys
sys.path.append(os.getcwd())

import torch
import random
import numpy as np
import cv2


seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


length_of_embedding=args.length_of_embedding

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)


#=================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(device=device, dataset_dir = f'{args.data_dir}', mode = 'train')
testing_dataset  = MyDataset(device=device, dataset_dir = f'{args.data_dir}', mode = 'test')

train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(testing_dataset,  batch_size=32, shuffle=False)
#========================================================
print('dataloaded')
#=================== import Network =====================
from src.Network import Generator
model_Generator = Generator(length_of_embedding=length_of_embedding, model = args.model)
model_Generator.to(device)
#========================================================
print('generator loaded')
#=================== import Loss ========================
# ***** VGGPerceptualLoss
from src.loss.PerceptionLoss import VGGPerceptualLoss
Perceptual_loss=VGGPerceptualLoss()
Perceptual_loss.to(device)

# ***** SSIM_Loss
from src.loss.SSIMLoss import SSIM_Loss
ssim_loss = SSIM_Loss()
ssim_loss.to(device)

# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss
ID_loss = ID_Loss(model = args.model, prot_scheme=args.prot_scheme,same_r=args.same_r , device=device, centered=args.centered)

# ***** Other losses
MAE_loss = torch.nn.L1Loss()
MSE_loss = torch.nn.MSELoss()
BCE_loss = torch.nn.BCELoss()
#========================================================
print('losses loaded')

#=================== Optimizers =========================
# ***** optimizer_Generator
optimizer_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-3)
scheduler_Generator = torch.optim.lr_scheduler.StepLR(optimizer_Generator, step_size=10, gamma=0.5)
#========================================================

# ---- loss weights -------------------------------------
# Tuned for an identity-reconstruction attack: down-weight SSIM (it biases toward
# blurry, structurally-plausible faces) and up-weight perceptual + identity vs the
# original: MAE + 0.75*ssim  + 0.02*Perceptual  + 0.025*ID 
W_SSIM, W_PERC, W_ID = 0.75, 0.05, 0.05

# ---- mixed precision (AMP) ----------------------------
# ~2x faster epochs / lower memory. GradScaler guards fp16 under/overflow.
amp_enabled = False   # AMP disabled -> full fp32 (matches the reference; avoids fp16 instability)
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

# ---- attack-success-rate (ASR) eval -------------------
# Reconstruction "succeeds" when the recon image, re-embedded by the *unprotected*
# facenet recognizer, verifies against the genuine template: unit-norm euclidean_l2
# below the facenet threshold. This is the real attack objective (loss alone hides it).
# NB: use a DEDICATED unprotected recognizer -- NOT ID_loss.FaceRecognition_transformer,
# which now runs the PROTECTED pipeline (its sketch makes the metric meaningless).
from model import InceptionResnetV1 as _ASRRecognizer
ASR_THRESHOLD = 1.1027
recognizer = _ASRRecognizer(pretrained='vggface2', device=device).eval()
#========================================================



#=================== Save models and logs ===============
import os
_run_tag = f'{args.prot_scheme}{"_centered" if args.centered else ""}_{args.same_r}'
training_files = f'training_files/{args.model}/{_run_tag}'
os.makedirs(f'{training_files}',exist_ok=True)
os.makedirs(f'{training_files}/models',exist_ok=True)
os.makedirs(f'{training_files}/Generated_images',exist_ok=True)
os.makedirs(f'{training_files}/logs_train',exist_ok=True)

with open(f'{training_files}/logs_train/generator.csv','w') as f:
    f.write("epoch,MAE_loss_Gen,ID_loss_Gen,Perceptual_loss_Gen,ssim_loss_Gen_test,total_loss_Gen,asr\n")

with open(f'{training_files}/logs_train/log.txt','w') as f:
    pass


for embedding, real_image, id in test_dataloader:
    pass
real_image=real_image.cpu()
for i in range(real_image.size(0)):
    #if i >2: 
    #    break
    os.makedirs(f'{training_files}/Generated_images/{i}', exist_ok=True)
    img = real_image[i].squeeze()
    # im = (img.numpy().transpose(1,2,0)); im*=128.0; im+=127.5; im = im.astype(int)
    im = (img.numpy().transpose(1,2,0)*255).astype(int)
    # print(min(im.flatten()), max(im.flatten()))
    # cv2.imwrite(f'{training_files}/Generated_images/{i}/real_image.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
    cv2.imwrite(f'{training_files}/Generated_images/{i}/real_image.jpg',im)
#========================================================
print('starting training')
#=================== Train ==============================
num_epochs=200
for epoch in range(num_epochs):  
    iteration=0
    
    print(f'epoch: {epoch}, \t learning rate: {optimizer_Generator.param_groups[0]["lr"]}')
    model_Generator.train()
    for embedding, real_image, id in train_dataloader:
        # ==================forward==================
        optimizer_Generator.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            generated_image = model_Generator(embedding)
            MAE = MAE_loss(generated_image, real_image)
            ID  = ID_loss(generated_image, real_image, id)  # protected pipeline needs per-id img_id
            Perceptual  = Perceptual_loss(generated_image, real_image)
            ssim = ssim_loss(generated_image, real_image)
            total_loss_Generato = MAE + W_SSIM*ssim + W_PERC*Perceptual + W_ID*ID
        # ==================backward (AMP-scaled)====
        scaler.scale(total_loss_Generato).backward()
        scaler.step(optimizer_Generator)
        scaler.update()
        # ==================log======================
        iteration +=1
        if iteration % 200 == 0:
            # model_Generator.eval()
            # print(f'epoch:{epoch+1} \t, iteration: {iteration}, \t total_loss:{total_loss_Generato.data.item()}')
            with open(f'{training_files}/logs_train/log.txt','a') as f:
                f.write(f'epoch:{epoch+1}, \t iteration: {iteration}, \t total_loss:{total_loss_Generato.data.item()}\n')
            pass
        
    # ******************** Eval Genrator ********************
    model_Generator.eval()
    MAE_loss_Gen_test = ID_loss_Gen_test = Perceptual_loss_Gen_test = ssim_loss_Gen_test = total_loss_Gen_test = 0
    asr_succ = asr_tot = 0
    iteration =0
    for embedding, real_image, id in test_dataloader:
        iteration +=1
        # ==================forward==================
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            generated_image = model_Generator(embedding)
            MAE = MAE_loss(generated_image, real_image)
            ID  = ID_loss(generated_image, real_image, id)  # protected pipeline needs per-id img_id
            Perceptual  = Perceptual_loss(generated_image, real_image)
            ssim = ssim_loss(generated_image, real_image)
            total_loss_Generato = MAE + W_SSIM*ssim + W_PERC*Perceptual + W_ID*ID
            ####
            MAE_loss_Gen_test += MAE.item()
            ID_loss_Gen_test  += ID.item()
            Perceptual_loss_Gen_test += Perceptual.item()
            ssim_loss_Gen_test  += ssim.item()
            total_loss_Gen_test += total_loss_Generato.item()
            # ---- ASR: recon re-embedded must verify vs the genuine template ----
            e_gen  = recognizer(generated_image * 2 - 1)
            e_real = recognizer(real_image * 2 - 1)
            e_gen  = e_gen  / e_gen.norm(dim=1, keepdim=True).clamp_min(1e-12)
            e_real = e_real / e_real.norm(dim=1, keepdim=True).clamp_min(1e-12)
            d = (e_gen - e_real).norm(dim=1)
            asr_succ += int((d < ASR_THRESHOLD).sum().item())
            asr_tot  += int(d.shape[0])

    with open(f'{training_files}/logs_train/generator.csv','a') as f:
        _asr = asr_succ / max(asr_tot, 1)
        f.write(f"{epoch+1}, {MAE_loss_Gen_test/iteration}, {ID_loss_Gen_test/iteration}, {Perceptual_loss_Gen_test/iteration}, {ssim_loss_Gen_test/iteration}, {total_loss_Gen_test/iteration}, {_asr}\n")
    print(f'epoch {epoch+1}: test total_loss={total_loss_Gen_test/iteration:.4f}  ASR={asr_succ/max(asr_tot,1):.4f}')
        
    generated_image = model_Generator(embedding).detach().cpu()
    for i in range(generated_image.size(0)):
        #if i >2: 
        #    break
        img = generated_image[i].squeeze()
        # im = (img.numpy().transpose(1,2,0)); im*=128.0; im+=127.5; im = im.astype(int)
        im = (img.numpy().transpose(1,2,0)*255).astype(int)
        # cv2.imwrite(f'{training_files}/Generated_images/{i}/epoch_{epoch+1}.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
        cv2.imwrite(f'{training_files}/Generated_images/{i}/epoch_{epoch+1}.jpg',im)
    # *******************************************************
    
    # Save model_Generator
    print(f'Saving model_Generator at epoch {epoch+1}')
    if epoch%100 == 0 or epoch == num_epochs-1:
        torch.save(model_Generator.state_dict(), f'{training_files}/models/Generator_{epoch+1}.pth')

    # Update optimizer_Generator lr
    scheduler_Generator.step()
#========================================================
