import glob
import os
from deepface import DeepFace
import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='celeba', help='Dataset name', choices=['celeba', 'lfw', 'casia_webface'])
args.add_argument('--model', type=str, default='Facenet512', help='model name and path to write images', choices=['Facenet512', 'ArcFace', 'VGG-Face'])
args.add_argument('--normalisation', type=str, default='ArcFace', help='normalisation procedure', choices=['ArcFace', 'base'])
args = args.parse_args()
# convert to lower case and remove numbers
model_folder = args.model.lower().replace('-', '_').replace('512', '')
path = f'../datasets/pp/{args.dataset}/'
write_path = f'{path}embeddings_{model_folder}/'
os.makedirs(write_path, exist_ok=True)
images = glob.glob(f'{path}images/*')
images.sort()

i = 0
for file in images:
    i += 1
    if (i % 100 == 0): print(i, flush=True)
    if os.path.exists(f'{write_path}' + file.split('/')[-1].replace('.jpg', '.npy')):
        print('Skipping', file.split('/')[-1], flush=True)
        continue
    embedding = DeepFace.represent(img_path=file, model_name=args.model, enforce_detection=False, detector_backend='skip', normalization='ArcFace') # normalisation procedure = (img -127.5 )/ 128.0
    embedding = embedding[0]['embedding']
    np.save(f'{write_path}' + file.split('/')[-1].replace('.jpg', '.npy'), embedding)

