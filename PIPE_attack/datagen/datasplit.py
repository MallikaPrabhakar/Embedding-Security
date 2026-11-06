# separates data into train set (1 image per person for all persons - 1000), and test set (1 image for remaining 1000 persons)

import os
import shutil
import argparse
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='celeba', help='Dataset name', choices=['celeba', 'lfw', 'casia_webface', 'celeba_128'])
args = args.parse_args()


def split_dataset(img_id_map, dataset):
    models = ['facenet']
    folders = ['train', 'test', 'test_seen', 'test2']
    for folder in folders:
        os.makedirs(f'../datasets/{dataset}/{folder}/images', exist_ok=True)
        for model in models:
            os.makedirs(f'../datasets/unprotected/{dataset}/{model}/{folder}/embeddings', exist_ok=True)

    for i, row in img_id_map.iterrows():
        img, id, label = row['img'], row['id'], row['label']
        shutil.copy(f'../../datasets/pp/{dataset}/images/{img}', f'../datasets/{dataset}/{label}/images/{img}')
        for model in models:
            shutil.copy(f'../../datasets/pp/{dataset}/embeddings_{model}/{img.split(".")[0]}.npy', f'../datasets/unprotected/{dataset}/{model}/{label}/embeddings/{img.split(".")[0]}.npy')

    
if __name__ == '__main__':
    img_id_map = pd.read_csv(f'../../datasets/pp/{args.dataset}/img_id_map.csv')
    split_dataset(img_id_map, args.dataset)
    shutil.copy(f'../../datasets/pp/{args.dataset}/img_id_map.csv', f'../datasets/{args.dataset}/img_id_map.csv')
    print('done')
