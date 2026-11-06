# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import bob.bio.base

impostors, genuines = bob.bio.base.score.load.split_csv_scores('results/scores-dev.csv')
_, invertes = bob.bio.base.score.load.split_csv_scores('results/scores_inversion-dev.csv')

import numpy as np
def fmr_fnmr(neg, pos, threshold):
    fmr, fnmr = 0, 0
    fmr = np.mean(np.where(np.array(neg) >= threshold,1,0))
    fnmr = np.mean(np.where(np.array(pos) < threshold,1,0))
    return fmr, fnmr
from bob.measure import far_threshold

for thr in [1e-2,1e-3]:
    print(f'======= threshold: {thr} =======')
    threshold = far_threshold(impostors, genuines, far_value=thr)

    fmr,fnmr = fmr_fnmr(impostors, genuines, threshold)
    _,uSAR   = fmr_fnmr(impostors, invertes, threshold)
            
    TMR = 1-fnmr
    SAR = 1-uSAR

    print(f'TMR and SAR: {round(TMR*100,2)} & {round(SAR*100,2)}')