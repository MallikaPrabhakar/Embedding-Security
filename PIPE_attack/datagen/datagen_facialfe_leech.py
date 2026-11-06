# Implementation for leech lattice based FE presented in the FacialFE paper.
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
from leech_helper import (Q, _preprocess, _to_int, _decode_vector_int, _to_real,)

# following pinwheel scales
FACENET_SCALE = 1461.76 


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Leech-lattice fuzzy-extractor embeddings')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--folders', nargs='+', default=None,
                        help='splits to process (default: train test_seen test test2)')
    return parser.parse_args()


def ss_gen(w: np.ndarray, scale: float = None, id: int = None):
    """Secure sketch (baseline): s = (c - w_int) mod Q where c = decode(w_int)."""
    w_int = _to_int(_preprocess(w), scale=scale)
    c = _decode_vector_int(w_int)[0]
    s = (c - w_int) % Q
    return s, c


def ss_rec(wp: np.ndarray, s: np.ndarray, scale: float = None) -> np.ndarray:
    """Recover the codeword from a noisy probe `wp` and sketch `s`."""
    wp_int = _to_int(_preprocess(wp), scale=scale)
    cp = (wp_int + s) % Q
    cp = _decode_vector_int(cp)[0]
    return cp


def main():
    args = parse_args()
    dataset = args.dataset
    model = args.model
    scale = FACENET_SCALE

    input_folder = f'../datasets/unprotected/{dataset}/{model}'
    img_id_map = pd.read_csv(f'../datasets/{dataset}/img_id_map.csv')
    folders = args.folders if args.folders is not None else ['train', 'test_seen', 'test', 'test2']

    for folder in folders:
        output_folder = f'../datasets/facialfe_leech/{dataset}/{model}_scale{scale}/{folder}'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f'{output_folder}/sketches', exist_ok=True)
        os.makedirs(f'{output_folder}/codewords', exist_ok=True)
        os.makedirs(f'{output_folder}/embeddings', exist_ok=True)
        os.makedirs(f'{output_folder}/embeddings_not_centered', exist_ok=True)

        files = glob.glob(f'{input_folder}/{folder}/embeddings/*.npy')
        print(f'Processing {len(files)} files in {input_folder}/{folder}')
        for file in tqdm(files):
            w = np.load(file)
            file_name = os.path.basename(file)[:-4]
            id = int(img_id_map[img_id_map['img'] == f'{file_name}.jpg']['id'].iloc[0])
            s, c = ss_gen(w, scale=scale)
            np.save(f'{output_folder}/sketches/{file_name}.npy', s)
            np.save(f'{output_folder}/codewords/{file_name}.npy', c)
            emb = _to_real((s + Q // 2) % Q)  # mod-Q center first so the embedding is a single bump, not U-shaped
            emb_not_centered = _to_real(s)
            np.save(f'{output_folder}/embeddings/{file_name}.npy', emb)
            np.save(f'{output_folder}/embeddings_not_centered/{file_name}.npy', emb_not_centered)
    return

if __name__ == '__main__':
    main()
