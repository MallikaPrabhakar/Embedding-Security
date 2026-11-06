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

min_ = np.min(impostors)
max_ = np.max(impostors)

all_x = []
all_y1 = []
all_y2 = []
for threshold in np.linspace(min_,max_,num=10000):
    fmr,fnmr = fmr_fnmr(impostors, genuines, threshold)
    _,uSAR   = fmr_fnmr(impostors, invertes, threshold)
    
    x = fmr
    y = 1-uSAR
    
    all_x.append(x)
    all_y.append(y)



from  matplotlib import pyplot as plt
plt.plot(all_x, all_y, label='SAR')

plt.legend()
plt.xscale('log')
plt.xlabel('FMR')
plt.ylabel('SAR')
plt.ylim(0,1.01)
plt.savefig('SAR_FMR.pdf')