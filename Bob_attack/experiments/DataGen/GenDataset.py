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
parser = argparse.ArgumentParser(description='Generate training dataset for face reconstruction')
parser.add_argument('--prot_scheme', metavar='<prot_scheme>', type= str, default='unprotected',
                    help='protection scheme to be used')
parser.add_argument('--same_r', metavar='<same_r>', type= str, default='none',
                    help='takes values, none, true, false')
parser.add_argument('--dataset', metavar='<dataset>', type= str, default='celeba',
                    help='dataset to be used')
parser.add_argument('--model', metavar='<model>', type= str, default='facenet',
                    help='model to be used')
parser.add_argument('--mode', metavar='<mode>', type= str, default='test',
                    help='generate from train set or test set')
parser.add_argument('--centered', action='store_true',
                    help='facialfe_leech / facialfe_leech_randomised: emit the '
                         'mod-q centered helper (_to_real((s+Q//2)%%Q)) instead of the raw value')
args = parser.parse_args()


import torch

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from model import InceptionResnetV1PipeFE

# PIPE_attack fuzzy-extractor schemes (recompute embeddings from images, facenet only),
# reproducing PIPE_attack/datasets/<scheme>/<dataset>/facenet_* exactly.
PIPE_FE_SCHEMES = ['unprotected', 'facialfe_e8', 'facialfe_leech',
                   'facialfe_leech_randomised', 'ourfe']
# schemes whose public helper depends on a per-identity secret codeword / keys -> need img_id
PIPE_SEEDED_SCHEMES = ['facialfe_leech_randomised', 'ourfe']

import cv2
import os
import glob
import random
import numpy as np
import time

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_FaceRecognition_transformer(model='facenet', prot_scheme='unprotected', same_r = 'none', device='cpu', dataset='celeba', centered=False):
    # The 5 PIPE_attack fuzzy-extractor schemes (incl. unprotected) all go through the
    # single InceptionResnetV1PipeFE wrapper: facenet backbone (last_bn) + the matching PIPE FE transform
    if prot_scheme in PIPE_FE_SCHEMES:
        if model != 'facenet':
            raise NotImplementedError(f"PIPE scheme '{prot_scheme}' is only wired for facenet")
        print(prot_scheme, same_r, 'centered' if centered else '')
        return InceptionResnetV1PipeFE(prot_scheme, same_r=same_r, pretrained='vggface2', device=device, dataset=dataset, centered=centered)

    raise ValueError(f"Unsupported prot_scheme '{prot_scheme}' (artifact supports: {PIPE_FE_SCHEMES}, facenet only)")


class Generate_Dataset():
    def __init__(self, data_dir = 'celeba_facenet_pp',
                        model = 'facenet',
                        mode = 'train',
                        device='cpu',
                        prot_scheme='unprotected',
                        same_r='none',
                        dataset='celeba',
                        centered=False
                ):
        self.device=device
        self.dir_all_images = []
        self.model = model
        self.mode = mode
        self.prot_scheme = prot_scheme
        self.same_r = same_r
        self.dataset = dataset
        self.data_dir = data_dir
        self.centered = centered
        
        folder = os.path.join(data_dir, mode, "images")
        print(folder)
        all_imgs = glob.glob(folder+'/*') # can be png or jpg
        all_imgs.sort()
        if mode == 'test': # pick 1000 only
            all_imgs = all_imgs[:1000]
        
        for img in all_imgs:
            self.dir_all_images.append(img)
        self.Face_Recognition_Network = get_FaceRecognition_transformer(model=model, prot_scheme=prot_scheme, same_r = same_r, device=self.device, dataset=dataset, centered=centered).eval()

    
    def generate_dataset(self, save_dir):
        os.makedirs(f'{save_dir}/images',exist_ok=True)
        os.makedirs(f'{save_dir}/embeddings',exist_ok=True)
        for idx in range(len(self.dir_all_images)):
            if (idx+1)%100 == 0:
                print(idx+1)
            embedding, image = self.__getitem__(idx)
            np.save(f'{save_dir}/images/{self.dir_all_images[idx][-10:-4]}', image)
            np.save(f'{save_dir}/embeddings/{self.dir_all_images[idx][-10:-4]}', embedding)

    def __getitem__(self, idx):

        image = cv2.imread(self.dir_all_images[idx]) # (160, 160, 3) already cropped and scaled
        # print(min(image.flatten()), max(image.flatten()))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img -127.5
        img /= 128.0
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        img_name = self.dir_all_images[idx].split('/')[-1]
        img_name = img_name.replace('.npy', '.jpg')

        if self.same_r == 'false' or self.prot_scheme in PIPE_SEEDED_SCHEMES:
            embedding = self.Face_Recognition_Network(img, img_id = np.array([img_name]) )
        else:
            embedding = self.Face_Recognition_Network(img)
        # embedding = self.Face_Recognition_Network.transform( (image*255.).astype('uint8') )
        # print(min(image.flatten()), max(image.flatten()))
        image = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        
        return embedding, image
    
    def transform_image(self,image):

        image = image.transpose(2,0,1) # (3, 160, 160)
        return image
    
    def transform_embedding(self,embedding):
        embedding = embedding.detach().cpu().numpy()
        embedding = np.reshape(embedding,[embedding.shape[-1],1,1])
        return embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

t = time.time()
read_dir = '../../../PIPE_attack/datasets/'+ args.dataset
DB_gen = Generate_Dataset(model = args.model, data_dir = read_dir, mode = args.mode, prot_scheme=args.prot_scheme, same_r=args.same_r, device=device, dataset=args.dataset, centered=args.centered)
# centered variants get a distinct "<scheme>_centered_<same_r>" tag so they don't clobber the raw ones
_tag = f'{args.prot_scheme}{"_centered" if args.centered else ""}_{args.same_r}'
DB_gen.generate_dataset(save_dir=f'./databases/{args.dataset}/{args.model}/{_tag}/{args.mode}')
print("Time taken to generate dataset:", time.time() - t)