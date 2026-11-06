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
import numpy as np

import sys
sys.path.insert(0, '../../../')
from paths import root_dir
sys.path.insert(0, f"{root_dir}/Bob_attack/experiments/")
from model import InceptionResnetV1PipeFE


# The PIPE_attack fuzzy-extractor schemes (incl. unprotected). The ID loss runs the
# generated and real images through the PROTECTED pipeline (backbone + FE), measuring
# identity in the protected-template space. The FE uses the differentiable straight-through
# version (pipe_fe_torch) so the loss backprops to the generator. ourfe has no surrogate
# -> its protected embedding is non-differentiable, so its ID term contributes no gradient.
PIPE_FE_SCHEMES = ['unprotected', 'facialfe_e8', 'facialfe_leech',
                   'facialfe_leech_randomised', 'ourfe']

def get_FaceRecognition_transformer(model='facenet', prot_scheme='unprotected', same_r = 'none', device='cpu', dataset='celeba', centered=False):
    if prot_scheme in PIPE_FE_SCHEMES:
        if model != 'facenet':
            raise NotImplementedError(f"PIPE scheme '{prot_scheme}' ID loss is only wired for facenet")
        print(prot_scheme, '-> ID loss uses the PROTECTED pipeline (differentiable STE FE; ourfe: no grad)')
        return InceptionResnetV1PipeFE(prot_scheme, same_r=same_r, pretrained='vggface2', device=device,
                                       dataset=dataset, centered=centered)

    raise ValueError(f"Unsupported prot_scheme '{prot_scheme}' (artifact supports: {PIPE_FE_SCHEMES}, facenet only)")


class ID_Loss:
    def __init__(self, model = 'facenet', prot_scheme='unprotected',same_r='none' , device='cpu', dataset='celeba', centered=False ):
        self.FaceRecognition_transformer = get_FaceRecognition_transformer(model=model, prot_scheme=prot_scheme,same_r=same_r, device=device, dataset=dataset, centered=centered).eval()
        self.prot_scheme = prot_scheme
        self.model = model
        self.same_r = same_r
        self.device = device
        # PIPE schemes feed the protected pipeline (the seeded ones -- facialfe_leech_randomised,
        # ourfe -- need the per-identity img_id); legacy same_r=='false' also needs it.
        self.use_img_id = (prot_scheme in PIPE_FE_SCHEMES) or (same_r == 'false')

    def __call__(self, img1, img2, id=None):
        # Note: input img should be in (-1,1)
        # originally, img1 img2 in [0,1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        if self.use_img_id:
            id = np.array(id)
            embedding1 = self.FaceRecognition_transformer(img1, img_id = id)
            embedding2 = self.FaceRecognition_transformer(img2, img_id = id)
        else:
            embedding1 = self.FaceRecognition_transformer(img1)
            embedding2 = self.FaceRecognition_transformer(img2)
        # In the PROTECTED space the per-id codeword is a large common component, so cosine
        # saturates (~1, dead gradient). MSE measures (e1-e2)^2, where that common offset
        # CANCELS in the difference -> clean, non-saturating identity signal.
        # return torch.nn.L1Loss()(embedding1,embedding2)-torch.nn.CosineSimilarity()(embedding1,embedding2).mean()
        return torch.nn.MSELoss()(embedding1, embedding2)