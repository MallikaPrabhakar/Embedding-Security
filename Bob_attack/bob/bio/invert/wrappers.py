# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
from sklearn.pipeline import Pipeline


def get_invert_pipeline(FR_transformer, inv_transformer, feature_extractor):

    # pipeline = make_pipeline(
    #    *[item for item in FR_transformer],
    #    inv_transformer,
    #    *[item for item in FR_transformer]
    # )
    pipeline = Pipeline(
        FR_transformer.steps
        + [
            ("inverted-samples", inv_transformer),
            ("inverted-features", feature_extractor),
        ]
    )

    return pipeline
