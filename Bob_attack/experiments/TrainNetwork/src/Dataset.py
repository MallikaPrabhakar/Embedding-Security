# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import torch
from torch.utils.data import Dataset
import glob
import random
import numpy as np

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, dataset_dir = './databases/celeba/facenet/unprotected_none',
                       mode='test',
                       device='cpu',
                       random_seed=2021
                ):
        self.dataset_dir = dataset_dir
        self.device = device
        self.mode = mode

        self.dir_all_images = []
        all_files = glob.glob(dataset_dir+'/'+mode+'/images/*')
        for file in all_files:
            file = file.split('/')[-1]
            self.dir_all_images.append(file)
        self.dir_all_images.sort()
                
            
    def __len__(self):
        return len(self.dir_all_images)

    def __getitem__(self, idx):

        image = f"{self.dataset_dir}/{self.mode}/images/{self.dir_all_images[idx]}"
        embedding = f"{self.dataset_dir}/{self.mode}/embeddings/{self.dir_all_images[idx]}"
       
        image     = np.load(image) # range (0,1), shape (3, 160, 160) or (3, 112, 112) 
        # print(image.shape)
        embedding = np.load(embedding) # shape (512,1,1)
        
        image     = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        id =  self.dir_all_images[idx] # remove .npy from the name
        id = id.replace('.npy', '.jpg')
        return embedding, image, id
    
    def transform_image(self,image):
        # ======= RESIZE ========
        image = image/255.
        # image = image - 127.5
        # image = image / 128.0
        image = torch.Tensor(image).to(self.device)
        return image
    
    def transform_embedding(self, embedding):
        # embedding = np.reshape(embedding,[embedding.shape[-1],1,1])
        embedding = torch.Tensor(embedding).to(self.device)
        return embedding