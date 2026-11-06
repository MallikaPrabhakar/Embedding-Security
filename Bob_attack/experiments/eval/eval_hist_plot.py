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

from bob.measure import far_threshold, eer_threshold, min_hter_threshold
threshold = far_threshold(impostors, genuines, far_value=1e-3)



import matplotlib.pyplot as plt
# plt.figure(figsize=(8.5, 2.5), dpi=80)
plt.figure(figsize=(7.5, 2), dpi=80)
plt.hist(impostors, bins = 25, density=True, label="Impostors", color="C0", alpha = 0.8)
plt.hist(genuines, bins = 25, density=True, label="Genuines", color="C2", alpha = 0.8)
plt.hist(invertes, bins = 25, density=True, label="Inversion Attack", color="C1", alpha = 0.7)

plt.axvline(threshold, linestyle='--',   label=f"FMR=0.001", color="black")


handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2,3]
legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                fontsize=10.5, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0,
                fancybox=True, shadow=False, ncol=4).get_frame()

legend.set_edgecolor("black")
legend.set_linewidth(.75)

plt.xlabel('score', fontsize=12)
plt.ylabel("Probability density", fontsize=12)
plt.xlim(-1.3,0)
plt.savefig('hist.pdf',bbox_inches='tight')