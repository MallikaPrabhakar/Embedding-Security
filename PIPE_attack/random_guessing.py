'''
for arguments model dataset folders, perform random guessing in two modes: L2, Rep
L2 takes the unprotected embedding of person 1 and computes the L2 distance to the unprotected embedding of person 2, for all pairs (1,2). This is the baseline for random guessing in the embedding space. We report the average L2 distance across all pairs.
Rep takes the unprotected embedding of person 1 and performs Gen, it then takes the unprotected embedding of person 2 and performs Rep, for all pairs (1,2). The codeword from person 1 is compared to the codeword from person 2. The number of blocks to compare depends on the scheme (ourfe operates at >= 4 blocks, while facialfe_leech operates at 21 blocks).
scheme facialfe_e8 and randomfe also use the rep procedure.
For unprotected embeddings, Rep is same as L2.

Both L2 and Rep are leave-one-out over a fixed-seed `--rep_subset`-image subset (e.g. the
1000-image test set -> 1000*999 = 999000 cross-image attempts): EVERY image in the subset is
used as the enrollment (person 1) and scored against ALL cross-identity impostors (person 2)
in the subset. Each enrollment image i yields one per-enrollment value (its L2 mean distance /
its attack-success-rate over impostors); we report mean and stddev OVER enrollment images.

Run from PIPE_attack/datagen (the ../datasets relative paths resolve there), in the
`deepface`/`ddpm` env. All schemes are pure numpy -- randomfe's Rep uses the saved LLL basis
(datasets/randomfe/.../lll_basis.npy), no sage:
    python random_guessing.py --scheme facialfe_leech --dataset celeba --folders test
    python random_guessing.py --scheme randomfe       --dataset celeba --folders test
Note: Rep is O(N^2) decodes over the set -- prefer the test split for large datasets.
'''
import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd

# path bootstrap (additive, no chdir -> safe for importers): put PIPE_attack/datagen on sys.path
# for the flat `import leech_helper` / `import datagen_*` used here and inside the datagen_* modules.
# Data paths below stay ../datasets, so run this from PIPE_attack/datagen.
_DATAGEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datagen")
if _DATAGEN not in sys.path:
    sys.path.insert(0, _DATAGEN)

THRESHOLD = 1.1027          # Facenet512 / euclidean_l2 (matches scripts/evaluation.py)

# leech-family scale + block settings (mirrors the datagen modules / eval_protected.py)
LEECH_DIM = 24
LEECH_NBLOCKS = 21
LEECH_SCALE = {"facialfe_leech": 1461.76, "facialfe_leech_randomised": 1461.76}
LEECH_BLOCK_THRESHOLD = {"facialfe_leech": 21, "facialfe_leech_randomised": 21}
E8_INVSCALE = {"facenet": 7.25}   # w is divided by this (see datagen_facialfe_e8)
E8_DIM = 8

PROTECTED = ("facialfe_e8", "facialfe_leech", "facialfe_leech_randomised", "randomfe", "ourfe")


# ----------------------------- data loading ---------------------------------
def load_embeddings(dataset, model, folders):
    """Return (W [N,512] float64, ids [N] int) for the given splits' unprotected embeddings."""
    img_id_map = pd.read_csv(f"../datasets/{dataset}/img_id_map.csv")
    id_of = dict(zip(img_id_map["img"], img_id_map["id"]))
    W, ids = [], []
    for folder in folders:
        files = sorted(glob.glob(f"../datasets/unprotected/{dataset}/{model}/{folder}/embeddings/*.npy"))
        for f in files:
            name = os.path.basename(f)[:-4]
            W.append(np.load(f).astype(np.float64).flatten())
            ids.append(int(id_of[f"{name}.jpg"]))
    return np.array(W), np.array(ids)


def count_matching_blocks(x, y, dim):
    """#blocks of length `dim` that match between x and y. Exact float compare (norm < 1e-6),
    NOT an int64 cast -- E8 codewords are half-integers (E8 = D8 u D8+1/2), and casting to int64
    truncates them (0.5->0) into spurious matches. Leech/ourfe codewords are integers, unaffected."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return int(sum(np.linalg.norm(x[k * dim:(k + 1) * dim] - y[k * dim:(k + 1) * dim]) < 1e-6
                   for k in range(len(x) // dim)))


def summarize(values, label):
    """mean +/- stddev over enrollment images."""
    v = np.asarray(values, dtype=np.float64)
    return v.mean(), v.std(), len(v)


# ----------------------------- L2 baseline ----------------------------------
def l2_per_enrollment(W, ids):
    """Leave-one-out random guessing: each of the N images authenticates as every OTHER
    cross-identity image (N*(N-1) attempts in total; 1000 images -> 999000). For enrollment
    image i we report its mean impostor euclidean_l2 distance and its ASR (fraction of the
    cross-identity impostors with dist < THRESHOLD). Returns (dist_i, asr_i) over enrollments."""
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    n = len(Wn)
    dist_i = np.empty(n); asr_i = np.empty(n)
    for i in range(n):
        d = np.linalg.norm(Wn - Wn[i], axis=1)
        imp = d[ids != ids[i]]
        dist_i[i] = imp.mean()
        asr_i[i] = (imp < THRESHOLD).mean()
    return dist_i, asr_i


# ----------------------------- Rep baselines --------------------------------
def rep_leech_per_enrollment(scheme, W, ids):
    """For each enrollment i: Gen(i) -> (s_i,c_i); for every impostor j: Rep(j, s_i) -> cp;
    success if matching 24-d blocks >= block_threshold. Returns (asr_i, bt, nblocks)."""
    import importlib
    from leech_helper import _to_int, _preprocess, _decode_vector_int, Q
    mod = importlib.import_module({"facialfe_leech": "datagen_facialfe_leech",
                                   "facialfe_leech_randomised": "datagen_facialfe_leech_randomised"}[scheme])
    scale = LEECH_SCALE[scheme]
    bt = LEECH_BLOCK_THRESHOLD[scheme]
    n = len(W)
    wint = [_to_int(_preprocess(W[j]), scale=scale) for j in range(n)]   # precompute probe ints
    asr_i = np.empty(n)
    t0 = time.time()
    for i in range(n):
        s_i, c_i = mod.ss_gen(W[i], scale=scale, id=int(ids[i]))
        succ = []
        for j in range(n):
            if ids[j] == ids[i]:
                continue
            cp = _decode_vector_int((wint[j] + s_i) % Q)[0]
            succ.append(count_matching_blocks(c_i, cp, LEECH_DIM) >= bt)
        asr_i[i] = float(np.mean(succ))
        if (i + 1) % 100 == 0:
            print(f"  {scheme} rep enroll {i+1}/{n} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return asr_i, bt, LEECH_NBLOCKS


def rep_ourfe_per_enrollment(W, ids):
    """ourfe (our defense): decode-first keyed-hash sketch. For each enrollment i, Gen(i)
    yields (keys_i, c_i=hash(decode(w_i), keys_i)); for every impostor j we RECOMPUTE the
    keyed hash of the impostor's decoded codeword with person i's keys and count matching
    per-block hashes. Since the keyed hash is injective per block, a block matches iff the
    impostor's decoded codeword block equals i's. Success at >= 4 of 21 blocks."""
    from leech_helper import _to_int, _preprocess, _decode_vector_int, _vector_hash
    import datagen_ourdef as mod
    scale = mod.FACENET_SCALE
    bt = 4                                                      # decode-first operating point (>= 4 of 21 blocks)
    n = len(W)
    wdec = [_decode_vector_int(_to_int(_preprocess(W[j]), scale=scale))[0] for j in range(n)]
    asr_i = np.empty(n)
    t0 = time.time()
    for i in range(n):
        keys_i, c_i = mod.ss_gen(W[i], scale=scale, id=int(ids[i]))   # keys_i, hashes_i
        c_i = np.asarray(c_i)
        succ = []
        for j in range(n):
            if ids[j] == ids[i]:
                continue
            cp = np.asarray(_vector_hash(wdec[j], keys_i))           # impostor codeword, person-i keys
            succ.append(int(np.sum(cp == c_i)) >= bt)                # matching per-block hashes
        asr_i[i] = float(np.mean(succ))
        if (i + 1) % 100 == 0:
            print(f"  ourfe rep enroll {i+1}/{n} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return asr_i, bt, LEECH_NBLOCKS


def rep_e8_per_enrollment(model, W, ids):
    """For each enrollment i: s_i,c_i = Gen(w_i/t); for impostor j: c_rec = decode(w_j/t + s_i);
    success if all 8-d blocks match (full key). Returns (asr_i, bt, nblocks)."""
    import datagen_facialfe_e8 as e8
    t = E8_INVSCALE[model]
    n = len(W)
    nblocks = W.shape[1] // E8_DIM
    bt = nblocks
    asr_i = np.empty(n)
    t0 = time.time()
    for i in range(n):
        s_i, c_i = e8.ss_gen(W[i] / t)
        succ = []
        for j in range(n):
            if ids[j] == ids[i]:
                continue
            c_rec = e8.decode(W[j] / t + s_i)
            succ.append(count_matching_blocks(c_i, c_rec, E8_DIM) >= bt)
        asr_i[i] = float(np.mean(succ))
        if (i + 1) % 100 == 0:
            print(f"  e8 rep enroll {i+1}/{n} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return asr_i, bt, nblocks


_RANDOMFE_SHARED = "../datasets/randomfe/celeba/facenet_m180_n60_q130003_scale0.0014_sharedA"
_RANDOMFE_LLL_BASIS = f"{_RANDOMFE_SHARED}/lll_basis.npy"   # saved by datagen_randomfe.py


def rep_randomfe_per_enrollment(dataset, model, W, ids):
    """Same-A FOWH random-guessing -- fully NUMPY, no sage in the loop.

    With the SHARED A the enrolment codeword b_i CANCELS in the recovery, so Rep-success
    reduces to a pure lattice-membership test:
        success(i, j)  <=>  nearest_lattice(x_i - x_j) == 0 (mod q)   on L = [A | qI]
    (verified bit-identical to the sage solve-based key-recovery: 0/182 mismatches).
    x = process_input(P_proj @ e) reproduced in numpy; the nearest-plane uses the LLL basis
    saved by datagen_randomfe.py (lll_basis.npy). O(N^2) but vectorised over each
    enrolment's impostors, so the full split is feasible. A/P are shared across datasets.
    Returns (asr_i, 1, 1)."""
    import json
    A = np.load(f"{_RANDOMFE_SHARED}/A.npy").astype(np.int64)            # (m, n)   shared
    P = np.load(f"{_RANDOMFE_SHARED}/projection_matrix.npy")             # (m, 512) shared (P_proj)
    Bz = np.load(_RANDOMFE_LLL_BASIS).astype(np.float64)                 # (m, m) LLL basis of [A|qI]
    pr = json.load(open("randomfe_parameters.json"))["parameters"][2 if model == "facenet" else 3]
    m, q, scale, max_proj, min_proj = pr["m"], pr["q"], pr["scale"], pr["max_proj"], pr["min_proj"]

    # Gram-Schmidt of the LLL basis (numpy)
    G = np.zeros_like(Bz); G[0] = Bz[0]
    for k in range(1, Bz.shape[0]):
        G[k] = Bz[k].copy()
        for l in range(k):
            G[k] -= (Bz[k] @ G[l]) / (G[l] @ G[l]) * G[l]
    Gn = (G * G).sum(1)

    def zq_x(v):                                                         # == fowh.process_input(P_proj @ e)
        v = (v - min_proj) / (max_proj - min_proj) * q
        v = np.floor(v).astype(np.int64) % q
        if scale:
            v = v * scale
            v = np.floor(v).astype(np.int64) % q
        return v

    X = np.stack([zq_x(P @ W[j]) for j in range(len(W))]).astype(np.int64)     # (N, m) in Z_q  (P_proj @ e)

    def nearest_is_zero(D):                                              # D: (k, m) signed -> bool (k,)
        b = D.astype(np.float64).copy()                                 # Babai nearest-plane, batched
        for k in reversed(range(Bz.shape[0])):
            coef = np.rint((b @ G[k]) / Gn[k])                          # (k,) round-half-to-even
            b = b - np.outer(coef, Bz[k])
        v = np.rint(D - b).astype(np.int64)                             # nearest lattice point
        return (v % q == 0).all(axis=1)

    N = len(W); asr_i = np.empty(N); t0 = time.time()
    for i in range(N):
        asr_i[i] = float(nearest_is_zero(X[i] - X[ids != ids[i]]).mean())   # impostor success rate
        if (i + 1) % 200 == 0:
            print(f"  randomfe(numpy) enroll {i+1}/{N} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return asr_i, 1, 1


# --------------------------------- main -------------------------------------
CSV_COLUMNS = ["dataset", "model", "folders", "scheme", "threshold", "n_enroll",
               "l2_mean_dist", "l2_dist_std", "l2_asr", "l2_asr_std",
               "rep_n_enroll", "rep_block_threshold", "rep_nblocks", "rep_asr", "rep_asr_std"]


def write_csv(path, row):
    """Upsert `row` into CSV at `path` (replace any existing row with the same
    dataset/model/scheme/folders key), keeping a stable column order."""
    key = ["dataset", "model", "scheme", "folders"]
    new = pd.DataFrame([row], columns=CSV_COLUMNS)
    if os.path.exists(path):
        df = pd.read_csv(path)
        mask = (df[key] == pd.Series({k: row[k] for k in key})).all(axis=1)
        df = pd.concat([df[~mask], new], ignore_index=True)
    else:
        df = new
    df = df.sort_values(["dataset", "model", "scheme", "folders"]).reset_index(drop=True)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[csv] wrote {path} ({len(df)} rows)")


def parse_args():
    p = argparse.ArgumentParser(description="random-guessing baselines (L2 + Rep), exhaustive over enrollments")
    p.add_argument("--dataset", default="celeba")
    p.add_argument("--model", default="facenet")
    p.add_argument("--folders", nargs="+", default=["test"],
                   help="splits to load unprotected embeddings from (default: test)")
    p.add_argument("--scheme", default="unprotected", choices=["unprotected", *PROTECTED],
                   help="protection scheme for the Rep baseline (unprotected -> Rep == L2)")
    p.add_argument("--csv", default="random_guessing_results.csv",
                   help="CSV to load + upsert this run's results into (keyed by dataset/model/scheme/folders)")
    p.add_argument("--rep_subset", type=int, default=100,
                   help="both L2 and Rep are O(N^2) over the set; restrict to a fixed-seed subset of "
                        "this many images (-> rep_subset*(rep_subset-1) cross-image attempts).")
    p.add_argument("--rep_seed", type=int, default=0, help="fixed seed for the L2/Rep subset selection")
    return p.parse_args()


def pick_subset(W, ids, k, seed):
    """Deterministically pick min(k, N) image indices for the Rep baseline."""
    n = len(W)
    if k <= 0 or k >= n:
        return W, ids, n
    idx = np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))
    return W[idx], ids[idx], k


def main():
    args = parse_args()
    W, ids = load_embeddings(args.dataset, args.model, args.folders)
    Wr, idsr, sub_n = pick_subset(W, ids, args.rep_subset, args.rep_seed)   # one subset for L2 + Rep
    print(f"loaded {len(W)} embeddings ({W.shape[1]}-d) from {args.dataset}/{args.model} {args.folders}; "
          f"{len(set(ids.tolist()))} identities -> L2 & Rep use a {sub_n}-image subset (seed={args.rep_seed}) "
          f"-> up to {sub_n*(sub_n-1)} cross-image attempts")

    # ---- L2 baseline (always): leave-one-out, each image authenticates as every other ----
    dist_i, l2asr_i = l2_per_enrollment(Wr, idsr)
    l2_mean, l2_dist_std, _ = summarize(dist_i, "l2_dist")
    l2_asr, l2_asr_std, n_enroll = summarize(l2asr_i, "l2_asr")
    print(f"[L2]  n_enroll={n_enroll}  euclidean_l2 = {l2_mean:.4f} +/- {l2_dist_std:.4f}  "
          f"random-guess ASR(<{THRESHOLD}) = {l2_asr:.4f} +/- {l2_asr_std:.4f}")

    # ---- Rep baseline (per scheme): same subset, O(N^2) decodes, mean +/- std over enrollments ----
    folders_str = " ".join(args.folders)
    if args.scheme == "unprotected":
        rep_n, rep_bt, rep_nblk = n_enroll, "", ""
        rep_asr, rep_asr_std = l2_asr, l2_asr_std            # Rep == L2
        print(f"[Rep] scheme=unprotected -> Rep == L2: ASR = {rep_asr:.4f} +/- {rep_asr_std:.4f}")
    else:
        rep_n = sub_n
        print(f"[Rep] scheme={args.scheme}: {rep_n} enrollment imgs (subset seed={args.rep_seed}) "
              f"-> up to {rep_n*(rep_n-1)} pairs")
        if args.scheme in ("facialfe_leech", "facialfe_leech_randomised"):
            repasr_i, rep_bt, rep_nblk = rep_leech_per_enrollment(args.scheme, Wr, idsr)
        elif args.scheme == "ourfe":
            repasr_i, rep_bt, rep_nblk = rep_ourfe_per_enrollment(Wr, idsr)
        elif args.scheme == "facialfe_e8":
            repasr_i, rep_bt, rep_nblk = rep_e8_per_enrollment(args.model, Wr, idsr)
        elif args.scheme == "randomfe":
            repasr_i, rep_bt, rep_nblk = rep_randomfe_per_enrollment(args.dataset, args.model, Wr, idsr)
        rep_asr, rep_asr_std, _ = summarize(repasr_i, "rep_asr")
        print(f"[Rep] scheme={args.scheme}  rep_n_enroll={rep_n}  "
              f"random-guess ASR(>= {rep_bt}/{rep_nblk} blocks) = {rep_asr:.4f} +/- {rep_asr_std:.4f}")

    write_csv(args.csv, {
        "dataset": args.dataset, "model": args.model, "folders": folders_str,
        "scheme": args.scheme, "threshold": THRESHOLD, "n_enroll": n_enroll,
        "l2_mean_dist": round(l2_mean, 6), "l2_dist_std": round(l2_dist_std, 6),
        "l2_asr": round(l2_asr, 6), "l2_asr_std": round(l2_asr_std, 6),
        "rep_n_enroll": rep_n, "rep_block_threshold": rep_bt, "rep_nblocks": rep_nblk,
        "rep_asr": round(rep_asr, 6), "rep_asr_std": round(rep_asr_std, 6),
    })


if __name__ == "__main__":
    main()
