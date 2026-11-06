# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
from bob.bio.face.embeddings.pytorch import TF_NAS
from bob.bio.face.utils import lookup_config_from_database


annotation_type, fixed_positions, memory_demanding = lookup_config_from_database(
    locals().get("database")
)


def load(annotation_type, fixed_positions=None, memory_demanding=False):
    return TF_NAS(annotation_type, fixed_positions, memory_demanding)


pipeline = load(annotation_type, fixed_positions, memory_demanding)

