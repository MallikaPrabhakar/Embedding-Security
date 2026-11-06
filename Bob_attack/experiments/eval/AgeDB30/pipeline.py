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
parser = argparse.ArgumentParser(description='Vulnerability evaluation of face reocgnition system against template inversion attack')
parser.add_argument('--FR_system', metavar='<FR_system>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace')


import os
epoch=90
generator_checkpoint = f"{os.path.dirname( os.path.dirname(os.getcwd()) )}/TrainNetwork/training_files/models/Generator_{epoch}.pth"


# ================== dataset ======================
from load_db import get_database

# ================== Transformers ==================
if args.FR_system=="ArcFace":
    from bob.bio.face.embeddings.pytorch import iresnet100 as get_pipeline
elif args.FR_system=="ElasticFace":
    from bob.bio.face.embeddings.pytorch import iresnet100_elastic as get_pipeline
elif args.FR_system=='AttentionNet92':
    from bob.bio.facexzoo.transformers.pytorch import AttentionNet92 as get_pipeline
elif args.FR_system== 'HRNet':
    from bob.bio.facexzoo.transformers.pytorch import HRNet as get_pipeline
elif args.FR_system== 'RepVGG_B1':
    from bob.bio.facexzoo.transformers.pytorch import RepVGG_B1 as get_pipeline
elif args.FR_system== 'SwinTransformer_S':
    from bob.bio.facexzoo.transformers.pytorch import SwinTransformer_S as get_pipeline
else:
     print(f"[eval pipeline] {args.FR_system} is not defined!")


pipeline = get_pipeline(("eyes-center", fixed_positions=None, memory_demanding=False)

FR_transformer = pipeline.transformer

# ================== Inversion Transformer ===========
from bob.pipelines import wrap, CheckpointWrapper, SampleWrapper
from bob.bio.invert.wrappers import get_invert_pipeline


import os,sys
sys.path.append(os.path.dirname(os.getcwd()))
from transformers import InversionTransformer

inv_transformer = InversionTransformer(checkpoint=generator_checkpoint)


# The feature extractor is the last element of the pipeline
feature_extractor = FR_transformer[-1]
inversionAttack_transformer = get_invert_pipeline(
    FR_transformer, inv_transformer, feature_extractor
)


# ================== pipeline ======================
from bob.bio.invert.invertibility_pipeline import InvertBiometricsPipeline
from bob.bio.base.pipelines.vanilla_biometrics import Distance

algorithm = Distance()

invert_pipeline = InvertBiometricsPipeline(
    FR_transformer, inversionAttack_transformer, algorithm
)


from dask.distributed import Client
from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
dask_client = "single-threaded"
## DISABLE THIS TO USE THE SGE
# cluster = SGEMultipleQueuesCluster(min_jobs=1)
# dask_client = Client(cluster)

from bob.bio.invert.pipeline import execute_inverted_vanilla_biometrics
from tqdm import tqdm

num_folds=10
for f in tqdm(range(num_folds)):  
    print(f)  
    database = get_database(age_protocol=30,fold=f)

    execute_inverted_vanilla_biometrics(
        pipeline=invert_pipeline,
        database=database,
        dask_client=dask_client,
        groups=["dev"],
        output=f"./results/{f}",
        write_metadata_scores=True,
        checkpoint=False,
        dask_partition_size=200,
        dask_n_workers=0,
        dask_n_partitions=None,
    )

for csvfiles_to_merge in ['scores-dev','scores_inversion-dev']:
    final_csv_path=f'results/{csvfiles_to_merge}.csv'
    os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
    with open(final_csv_path, "w") as f:
        for i in range(num_folds):
            # Not memory intensive score writing
            with open(f'results/{i}/{csvfiles_to_merge}.csv', "r") as f1:
                if i > 0:
                    f1.readline()  # skip header line
                for line in f1:
                    f.write(line)