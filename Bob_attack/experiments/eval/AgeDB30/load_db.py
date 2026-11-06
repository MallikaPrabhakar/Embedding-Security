# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.io.base import load
from bob.pipelines import DelayedSample, Sample
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
import numpy as np
import os 

from clapper.rc import UserDefaults
rc = UserDefaults("bobrc.toml")
AgeDB_PATH = rc.get("bob.db.agedb.directory")  # path to AgeDB dataset

def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}")) 

def load_RGB(path,):
    data = load(path)
    if data.ndim==2:
        data = np.repeat(np.expand_dims(data,axis=0),3,axis=0)
    return data

sample_loader = CSVToSampleLoaderBiometrics(
    data_loader=load_RGB,
    dataset_original_directory=f"{AgeDB_PATH}/protocol/03_Protocol_Images",
    extension=".jpg",
    reference_id_equal_subject_id=False,
)

def read_annotation_file(file_name):
    if not file_name:
        return None
    
    if not os.path.exists(file_name):
        raise IOError("The annotation file '%s' was not found" % file_name)
    

    
    annotations = read_pts(file_name)

    annotations = {'leye':np.mean(annotations[42:47,:],axis=0)[::-1],
                   'reye':np.mean(annotations[36:41,:],axis=0)[::-1],
    }
        
    # print(file_name,annotations)

    if (
        annotations is not None
        and "leye" in annotations
        and "reye" in annotations
        and annotations["leye"][1] < annotations["reye"][1]
    ):
        print(
            "The eye annotations in file '%s' might be exchanged!" % file_name
        )

    return annotations


class AgeDBAnnotations(BaseEstimator):
    def __init__(self):
        self.annotation_directory=f"{AgeDB_PATH}/protocol/03_Protocol_Images"
        self.annotation_extension=".pts"

    def transform(self, X): 
        if self.annotation_directory is None:
            return None
        # set_trace()
        annotated_samples = []
        for x in X:

            # since the file id is equal to the file name, we can simply use it
            annotation_file = os.path.join(
                self.annotation_directory, x.key + self.annotation_extension
            )
            
            # annotated_samples.append(
            #     DelayedSample.from_sample(
            #         x,
            #         delayed_attributes=dict(
            #             annotations=lambda: read_annotation_file(
            #                 annotation_file
            #             )
            #         ),
            #     )
            # )

            annotated_samples.append(
                Sample(
                    data=x.data,
                    parent=x,
                    annotations = read_annotation_file(
                        annotation_file
                    )
                )
            )
        return annotated_samples


annotations_loader = AgeDBAnnotations()

sample_loader = make_pipeline(sample_loader, annotations_loader)


# database = CSVDataset(
#     name="AgeDB",
#     protocol="5year-fold0",
#     dataset_protocol_path="./",
#     csv_to_sample_loader=sample_loader,
#     score_all_vs_all=False,
#     is_sparse=True,
# )
# database.annotation_type = "eyes-center"


import scipy.io as sio
def get_database(age_protocol=5,fold=0):
    """
    Arguments:
        age_protocol (int): 5,10,20,30
        fold (int): 0,1,2,3,4,5,6,7,8,9
    """
    protocol = f'{age_protocol}year-fold{fold}'
    if age_protocol==5:
        matfile = f'01_FINAL_protocol_{age_protocol}_years.mat'
    elif age_protocol==10:
        matfile = f'02_FINAL_protocol_{age_protocol}_years.mat'
    elif age_protocol==20:
        matfile = f'03_FINAL_protocol_{age_protocol}_years.mat'
    elif age_protocol==30:
        matfile = f'04_FINAL_protocol_{age_protocol}_years.mat'
    else:
        print(f'No protocol for {age_protocol}!')
    data = sio.loadmat(f'{AgeDB_PATH}/protocol/{matfile}')

    models = data['splits'][fold][0][0][0][0][0]
    probes = data['splits'][fold][0][0][0][0][1]

    models_list =[]
    probes_list =[]

    for i,model in enumerate(models):
        filename= model[0][0][0][0]
        age = model[0][0][1][0][0]
        subject_id = filename.split('_')[1]
        models_list.append({"path":filename, "reference_id":f'{i}_{filename}',"subject_id":subject_id, "age":age})

    for i,probe in enumerate(probes):
        filename= probe[0][0][0][0]
        age = probe[0][0][1][0][0]
        subject_id = filename.split('_')[1]
        probes_list.append({"path":filename, "reference_id":f'{i}_{filename}',"compare_reference_id":models_list[i]['reference_id'],"subject_id":subject_id,"age":age})

    os.makedirs(f'AgeDB/{protocol}/dev',exist_ok=True)

    from csv import DictWriter
    with open(f'AgeDB/{protocol}/dev/for_models.csv','w') as f:
        w = DictWriter(f, ["path","reference_id","subject_id","age"])
        w.writeheader()
        for d in models_list:
            w.writerow(d)

    with open(f'AgeDB/{protocol}/dev/for_probes.csv','w') as f:
        w = DictWriter(f, ["path","reference_id","compare_reference_id","subject_id","age"])
        w.writeheader()
        for d in probes_list:
            w.writerow(d)

    
    database = CSVDataset(
        name="AgeDB",
        protocol=protocol,
        dataset_protocol_path="./",
        csv_to_sample_loader=sample_loader,
        # score_all_vs_all=False,
        allow_scoring_with_all_biometric_references=False,
        is_sparse=True,
    )
    database.annotation_type = "eyes-center"

    return database
