# Random-matrix (FOWH / random-linear-code, LWE-style) fuzzy-extractor datagen,
# structured like datagen_facialfe_leech_randomised.py, built on the FOWH reference
# construction, run in the `sage` environment:
#
#   # in-domain (celeba, all splits) -- default:
#     sage -python datagen_randomfe.py
#   # cross-dataset (test split only) for casia_webface / lfw:
#     sage -python datagen_randomfe.py --dataset casia_webface
#     sage -python datagen_randomfe.py --dataset lfw
#   # explicit dataset / folders / model:
#     sage -python datagen_randomfe.py --dataset celeba --folders train test --model facenet
#
# Config: SAME projection matrix P_proj and SAME LWE matrix A for everyone (fixed
# seeds), so a celeba-trained inversion model can also be evaluated on other
# datasets' protected embeddings; the secret b varies by identity.
#   P_proj : fixed (m,512) projection, seed 1740470466 -> saved once
#   A      : fixed random_matrix(GF(q), m, n)   (seed A_SEED)          -> saved once
#   per identity id:  b = random_vector(GF(q), n)   (seed = id)        <- the codeword
#   per sample:       x = process_input(P_proj @ e) in Z_q   [fowh]
#                     c = A*b + x  (= Ab+x)                   (m-d public sketch)
#
# Stored:
#   <root>/projection_matrix.npy , <root>/A.npy , <root>/lll_basis.npy   (shared, once)
#   <split>/codewords/  : b        (per identity)
#   <split>/p/          : Ab+x     (per sample; public sketch)
#
# lll_basis.npy is the LLL-reduced basis of [A | qI] (rows). Recovery/eval consumers Babai-decode
# with it (G/Gnorm recompute from B in <1s), skipping the one-time ~132s LLL. No embeddings are
# produced here -- the Voronoi-residual conditioning vector is not generated.
import os
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd


from sage.all import set_random_seed, random_matrix, random_vector, GF, identity_matrix, ZZ  # noqa: E402
from randomfe_helper import process_input, fast_lll                                                              # noqa: E402

PROJECTION_SEED = 1740470466  # fixed projection seed (shared across datasets)
A_SEED = 2840470466          # single shared A for everyone

# default splits per dataset: celeba has the full set; others (cross-dataset) only test.
DEFAULT_FOLDERS = {'celeba': ['train', 'test_seen', 'test', 'test2'], 'casia_webface': ['test'], 'lfw': ['test']}


def load_params(model):
    params = json.load(open('randomfe_parameters.json'))['parameters']
    idx = 2  # celeba facenet
    pr = params[idx]
    assert pr['model'] == model, 'wrong parameters index'
    return pr['m'], pr['n'], pr['q'], pr['scale'], pr['max_proj'], pr['min_proj']


def to_int_vec(v):
    return np.array([int(e) for e in v], dtype=np.int64)


def to_int_mat(A, m, n):
    return np.array([int(e) for e in A.list()], dtype=np.int64).reshape(m, n)


def parse_args():
    p = argparse.ArgumentParser(description='randomfe (FOWH) fuzzy-extractor datagen')
    p.add_argument('--dataset', default='celeba', help='dataset name (celeba, casia_webface, lfw, ...)')
    p.add_argument('--folders', nargs='+', default=None,
                   help='splits to process (default: celeba -> train,test_seen,test,test2; else -> test)')
    p.add_argument('--model', default='facenet', help='embedding model (facenet)')
    return p.parse_args()


def main():
    args = parse_args()
    DATASET, MODEL = args.dataset, args.model
    folders = args.folders if args.folders is not None else \
        DEFAULT_FOLDERS.get(DATASET)

    m, n, q, scale, max_proj, min_proj = load_params(MODEL)
    print(f'randomfe(FOWH): dataset={DATASET} model={MODEL} folders={folders}')
    print(f'  m={m} n={n} q={q} scale={scale} | shared A, b by id | exact Babai decode')

    # (m, 512) projection applied as P_proj @ e; projection_matrix =
    # np.random.RandomState(PROJECTION_SEED).randn(m,512)/sqrt(m).
    P_proj = np.random.RandomState(PROJECTION_SEED).randn(m, 512) / np.sqrt(m)
    set_random_seed(A_SEED)
    A = random_matrix(GF(q), m, n)
    A_np = to_int_mat(A, m, n)

    # ---- one-time lattice setup for the shared A: LLL-reduce [A | qI], saved as lll_basis.npy ----
    t0 = time.time()
    Mat = A.change_ring(ZZ).augment(q * identity_matrix(ZZ, m))   # m x (n+m)
    B = fast_lll(Mat.T).change_ring(ZZ)                            # LLL-reduced basis (rows)
    B_np = np.array([int(e) for e in B.list()], dtype=np.int64).reshape(B.nrows(), B.ncols())
    print(f'LLL once: {time.time()-t0:.1f}s, basis {B.nrows()}x{B.ncols()}')

    input_folder = f'../datasets/unprotected/{DATASET}/{MODEL}'
    img_id_map = pd.read_csv(f'../datasets/{DATASET}/img_id_map.csv')
    base = f'../datasets/randomfe/{DATASET}/{MODEL}_m{m}_n{n}_q{q}_scale{scale}_sharedA'
    os.makedirs(base, exist_ok=True)
    np.save(f'{base}/projection_matrix.npy', P_proj)
    np.save(f'{base}/A.npy', A_np)
    np.save(f'{base}/lll_basis.npy', B_np)   # LLL-reduced basis of [A|qI]; consumers Babai-decode with it

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
            x = process_input((P_proj @ e).copy(), q, max_proj, min_proj, scale)   # GF(q)^m

            if id not in b_cache:
                set_random_seed(id)
                b_cache[id] = random_vector(GF(q), n)
            b = b_cache[id]
            c = A * b + x                              # Ab + x  (GF(q)^m)
            c_np, b_np = to_int_vec(c), to_int_vec(b)
            np.save(f'{out}/p/{name}.npy', c_np)
            np.save(f'{out}/codewords/{name}.npy', b_np)
            if (k + 1) % 200 == 0:
                print(f'  {k+1}/{len(files)} ({(time.time()-start)/60:.1f} min)', flush=True)
    print(f'done {DATASET} in {(time.time()-start)/60:.2f} min')


if __name__ == '__main__':
    main()
