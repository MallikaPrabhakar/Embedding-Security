# Pure-NumPy randomfe (FOWH / LWE-style) fuzzy-extractor datagen -- needs NO `sage` env.
# Produces ONLY the raw FE values:
#   <split>/codewords/ : b      (per-identity secret, seeded by id)
#   <split>/p/         : Ab + x  (per-sample public sketch/helper)  in Z_q
# (no embeddings/ -- the Voronoi-residual conditioning vector is not generated, so no LLL/Babai
# is needed; the whole pipeline is plain numpy.)
#
# Config (mirrors datagen_randomfe.py): SAME projection P_proj and SAME LWE matrix A for everyone
# (fixed seeds), saved once; the secret b varies by identity.
#   P_proj : fixed (m,512) projection, seed PROJECTION_SEED  -> projection_matrix.npy
#   A      : fixed (m,n) matrix in Z_q,  seed A_SEED          -> A.npy
#   per identity id:  b = RandomState(id).randint(0,q,n)      (the codeword)
#   per sample:       x = process_input(P_proj @ e) in Z_q  
#                     c = (A @ b + x) mod q                  (m-d public sketch)
#
# NOTE: numpy RNG != sage RNG, so A/b differ bit-for-bit from a sage run -- but the scheme is
# identical and self-consistent (A.npy is saved alongside).
#
#   python datagen_randomfe_numpy.py --dataset celeba
#   python datagen_randomfe_numpy.py --dataset casia_webface     # cross-dataset (test split)
#   python datagen_randomfe_numpy.py --dataset lfw
import os
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd

PROJECTION_SEED = 1740470466   # fixed projection seed (shared across datasets)
A_SEED = 2840470466            # single shared A for everyone

DEFAULT_FOLDERS = {'celeba': ['train', 'test_seen', 'test', 'test2'],
                   'casia_webface': ['test'], 'lfw': ['test']}


def load_params(model):
    params = json.load(open('randomfe_parameters.json'))['parameters']
    pr = params[2]  # celeba facenet
    assert pr['model'] == model, 'wrong parameters index'
    return pr['m'], pr['n'], pr['q'], pr['scale'], pr['max_proj'], pr['min_proj']


def process_input(x, q, max_proj, min_proj, scale):
    """Quantise a real vector to Z_q (the FOWH zq_x quantisation)."""
    x = x.astype(np.float64).copy()
    x -= min_proj
    x /= (max_proj - min_proj)
    x *= q
    x = np.floor(x).astype(np.int64) % q
    if scale:
        x = np.floor(x * scale).astype(np.int64) % q
    return x


def parse_args():
    p = argparse.ArgumentParser(description='randomfe (FOWH) fuzzy-extractor datagen -- pure numpy, p/ + codewords/ only')
    p.add_argument('--dataset', default='celeba')
    p.add_argument('--folders', nargs='+', default=None,
                   help='splits (default: celeba -> train,test_seen,test,test2; else -> test)')
    p.add_argument('--model', default='facenet')
    return p.parse_args()


def main():
    args = parse_args()
    DATASET, MODEL = args.dataset, args.model
    folders = args.folders if args.folders is not None else DEFAULT_FOLDERS.get(DATASET)

    m, n, q, scale, max_proj, min_proj = load_params(MODEL)
    print(f'randomfe(FOWH, numpy): dataset={DATASET} model={MODEL} folders={folders}')
    print(f'  m={m} n={n} q={q} scale={scale} | shared A, b by id | p = Ab+x (no embeddings)')

    # shared, deterministic projection and shared A.
    P_proj = np.random.RandomState(PROJECTION_SEED).randn(m, 512) / np.sqrt(m)
    A_np = np.random.RandomState(A_SEED).randint(0, q, size=(m, n)).astype(np.int64)

    input_folder = f'../datasets/unprotected/{DATASET}/{MODEL}'
    img_id_map = pd.read_csv(f'../datasets/{DATASET}/img_id_map.csv')
    base = f'../datasets/randomfe/{DATASET}/{MODEL}_m{m}_n{n}_q{q}_scale{scale}_sharedA'
    os.makedirs(base, exist_ok=True)
    np.save(f'{base}/projection_matrix.npy', P_proj)
    np.save(f'{base}/A.npy', A_np)

    b_cache = {}
    start = time.time()
    for folder in folders:
        out = f'{base}/{folder}'
        for sub in ('p', 'codewords'):
            os.makedirs(f'{out}/{sub}', exist_ok=True)
        files = sorted(glob.glob(f'{input_folder}/{folder}/embeddings/*.npy'))
        print(f'Processing {len(files)} files in {input_folder}/{folder}', flush=True)
        for k, file in enumerate(files):
            name = os.path.basename(file)[:-4]
            id = int(img_id_map[img_id_map['img'] == f'{name}.jpg']['id'].iloc[0])
            e = np.load(file).flatten()
            x = process_input(P_proj @ e, q, max_proj, min_proj, scale)        # Z_q^m

            if id not in b_cache:
                b_cache[id] = np.random.RandomState(id).randint(0, q, size=n).astype(np.int64)
            b = b_cache[id]
            c = (A_np @ b + x) % q                                             # Ab + x  (Z_q^m)

            np.save(f'{out}/p/{name}.npy', c.astype(np.int64))
            np.save(f'{out}/codewords/{name}.npy', b.astype(np.int64))
            if (k + 1) % 500 == 0:
                print(f'  {k+1}/{len(files)} ({(time.time()-start)/60:.1f} min)', flush=True)
    print(f'done {DATASET} in {(time.time()-start)/60:.2f} min')


if __name__ == '__main__':
    main()
