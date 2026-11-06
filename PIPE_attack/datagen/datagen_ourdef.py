# use the strategy of decoding embedding first and then applying the fuzzy extractor
import glob
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
sys.path.insert(0, os.getenv('PROJECT_ROOT'))
sys.path.insert(0, os.path.join(os.getenv('PROJECT_ROOT'), 'PIPE_attack'))
from leech_helper import (DIM, _preprocess, _to_int, _decode_vector_int, _vector_hash,
                          ourfe_embedding, sample_ourfe_keys,)

# following pinwheel scales
FACENET_SCALE = 144.64


def parse_args():
    parser = argparse.ArgumentParser(description='Generate our defense fe decoded input embeddings')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--folders', nargs='+', default=None,
                        help='splits to process (default: train test_seen test test2)')
    parser.add_argument('--scale', type=float, default=FACENET_SCALE,
                        help='leech scale (default: FACENET_SCALE)')
    parser.add_argument('--scheme', type=str, default='ourfe',
                        help='output scheme dir under datasets/ (e.g. ourfe)')
    return parser.parse_args()


def ss_gen(w, scale=None, id=None):
    """Decode-first secure sketch with a keyed-hash sketch.

    Seed the per-block keys by IDENTITY (np.random.seed(id)) so the public helper
    is reproducible from an image -- Bob's GenDataset recomputes it through the
    network."""
    w_c = _decode_vector_int(_to_int(_preprocess(w), scale=scale))[0]
    if id is not None:
        np.random.seed(id)
    keys = sample_ourfe_keys(len(w_c) // DIM)
    c = _vector_hash(w_c, keys) # keyed hash of the decoded codeword is the sketch
    return keys, c


def ss_rec(wp, keys, scale=None):
    """Decode-first recovery"""
    wp_c = _decode_vector_int(_to_int(_preprocess(wp), scale=scale))[0]
    cp = _vector_hash(wp_c, keys)
    return cp


def main():
    args = parse_args()
    dataset = args.dataset
    model = args.model
    scale = args.scale if args.scale is not None else FACENET_SCALE

    input_folder = f'../datasets/unprotected/{dataset}/{model}'
    img_id_map = pd.read_csv(f'../datasets/{dataset}/img_id_map.csv')
    folders = args.folders if args.folders is not None else ['train', 'test_seen', 'test', 'test2']

    for folder in folders:
        output_folder = f'../datasets/{args.scheme}/{dataset}/{model}_scale{scale}/{folder}'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f'{output_folder}/sketches', exist_ok=True)
        os.makedirs(f'{output_folder}/codewords', exist_ok=True)
        os.makedirs(f'{output_folder}/embeddings', exist_ok=True)

        files = glob.glob(f'{input_folder}/{folder}/embeddings/*.npy')
        print(f'Processing {len(files)} files in {input_folder}/{folder}')
        for file in tqdm(files):
            w = np.load(file)
            file_name = os.path.basename(file)[:-4]
            id = int(img_id_map[img_id_map['img'] == f'{file_name}.jpg']['id'].iloc[0])
            # s is keys
            s, c = ss_gen(w, scale, id)
            np.save(f'{output_folder}/sketches/{file_name}.npy', s)
            np.save(f'{output_folder}/codewords/{file_name}.npy', c)
            # public helper conditioning vector: flatten + normalise [keys, hashes] -> 42-d in [-1,1)
            emb = ourfe_embedding(s, c)
            np.save(f'{output_folder}/embeddings/{file_name}.npy', emb)
    return


if __name__ == '__main__':
    main()