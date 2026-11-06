"""TPR / FPR on a split for a protection scheme, using build_pairs genuine (same-person)
and impostor (different-person) pairs.

A pair "verifies as same" when:
  * unprotected : euclidean_l2 of the L2-normalised embeddings < THRESHOLD (1.1027)
  * leech family: Rep(emb2, sketch=Gen(emb1)) recovers a codeword matching Gen(emb1)'s
                  codeword in >= block_threshold of the 21 leech blocks
  * facialfe_e8 : decode(emb2/t + sketch) matches Gen(emb1)'s codeword in all 8-d blocks
  * randomfe    : Babai key-recovery of b from (helper(emb1), emb2) equals b1 (exact)

TPR = fraction of genuine pairs that verify; FPR = fraction of impostor pairs that verify.
build_pairs mode: 'test' -> test + test2 splits, 'train' -> train + test_seen splits.

    python tpr_fpr.py --scheme facialfe_leech --dataset celeba --mode test  --samples 2000
    python tpr_fpr.py --scheme facialfe_leech --dataset celeba --mode train --samples 2000
    python tpr_fpr.py --scheme randomfe       --dataset celeba --mode test  --samples 2000

The file lives in PIPE_attack/ but borrows datagen's helpers (helper.build_pairs, random_guessing,
the per-scheme datagen_* modules), so it chdir's into datagen at startup; relative ../datasets
and randomfe_parameters.json paths resolve exactly as for the datagen scripts. All schemes are pure numpy.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATAGEN = os.path.abspath(os.path.join(HERE, "datagen"))
sys.path.insert(0, DATAGEN)
os.chdir(DATAGEN)
from helper import build_pairs          # noqa: E402
import random_guessing as rg            # noqa: E402  (THRESHOLD, LEECH_*, E8_*, count_matching_blocks)

PROTECTED = ("facialfe_e8", "facialfe_leech", "facialfe_leech_randomised", "randomfe", "ourfe")
OURFE_SCALE = {"ourfe": 144.64}   # our defense at its operating scale
OURFE_BLOCK_THRESHOLD = 4                            # decode-first, >= 4 of 21 keyed-hash blocks


def id_lookup(dataset):
    m = pd.read_csv(f"../datasets/{dataset}/img_id_map.csv")
    return dict(zip(m["img"], m["id"]))


def id_of_path(p, id_of):
    return int(id_of[f"{os.path.basename(p)[:-4]}.jpg"])


# ---- per-scheme "verify as same?" over a list of (path1, path2) pairs ----
def verify_unprotected(pairs, id_of):
    out = []
    for p1, p2 in pairs:
        e1 = np.load(p1); e2 = np.load(p2)
        e1 = e1 / np.linalg.norm(e1); e2 = e2 / np.linalg.norm(e2)
        out.append(np.linalg.norm(e1 - e2) < rg.THRESHOLD)
    return np.array(out)


def verify_leech(scheme, pairs, id_of):
    import importlib
    mod = importlib.import_module({"facialfe_leech": "datagen_facialfe_leech",
                                   "facialfe_leech_randomised": "datagen_facialfe_leech_randomised"}[scheme])
    scale = rg.LEECH_SCALE[scheme]; bt = rg.LEECH_BLOCK_THRESHOLD[scheme]
    out = []
    for p1, p2 in pairs:
        s1, c1 = mod.ss_gen(np.load(p1), scale=scale, id=id_of_path(p1, id_of))
        cp = mod.ss_rec(np.load(p2), s1, scale=scale)
        out.append(rg.count_matching_blocks(c1, cp, rg.LEECH_DIM) >= bt)
    return np.array(out)


def verify_ourfe(scheme, pairs, id_of):
    """ourfe (our defense): Gen(emb1) -> (keys, hashes c1); recompute the keyed hash of
    emb2's decoded codeword with emb1's keys (ss_rec) -> cp; verify if >= 4 of 21 per-block hashes
    match. The keyed hash is injective per block, so this is the decode-first leech recovery."""
    import datagen_ourdef as mod
    scale = OURFE_SCALE[scheme]
    out = []
    for p1, p2 in pairs:
        keys, c1 = mod.ss_gen(np.load(p1), scale=scale, id=id_of_path(p1, id_of))
        cp = mod.ss_rec(np.load(p2), keys, scale=scale)
        out.append(int(np.sum(np.asarray(c1) == np.asarray(cp))) >= OURFE_BLOCK_THRESHOLD)
    return np.array(out)


def verify_e8(model, pairs, id_of):
    import datagen_facialfe_e8 as e8
    t = rg.E8_INVSCALE[model]; nblk = 512 // rg.E8_DIM
    out = []
    for p1, p2 in pairs:
        s1, c1 = e8.ss_gen(np.load(p1) / t)
        c_rec = e8.decode(np.load(p2) / t + s1)
        out.append(rg.count_matching_blocks(c1, c_rec, rg.E8_DIM) >= nblk)
    return np.array(out)


def verify_randomfe(dataset, model, pairs, id_of):
    """Same-A FOWH verification -- fully NUMPY, no sage. With the shared A the enrolment
    codeword cancels, so verify-success reduces to a lattice-membership test:
        success(p1, p2)  <=>  nearest_lattice(x1 - x2) == 0 (mod q)   on L = [A | qI]
    (verified bit-identical to the sage solve-based key-recovery: 0/182 criterion + 0/240
    basis mismatches). x = process_input(P_proj @ e) in numpy; nearest-plane uses the LLL basis
    saved by datagen_randomfe.py (lll_basis.npy). A/P are shared across datasets."""
    import json
    RF = "../datasets/randomfe/celeba/facenet_m180_n60_q130003_scale0.0014_sharedA"  # shared A/P
    P = np.load(f"{RF}/projection_matrix.npy")                                          # (m, 512) P_proj
    Bz = np.load(f"{RF}/lll_basis.npy").astype(np.float64)                              # saved by datagen_randomfe.py
    pr = json.load(open("randomfe_parameters.json"))["parameters"][2 if model == "facenet" else 3]
    q, scale, max_proj, min_proj = pr["q"], pr["scale"], pr["max_proj"], pr["min_proj"]
    G = np.zeros_like(Bz); G[0] = Bz[0]
    for k in range(1, Bz.shape[0]):
        G[k] = Bz[k].copy()
        for l in range(k):
            G[k] -= (Bz[k] @ G[l]) / (G[l] @ G[l]) * G[l]
    Gn = (G * G).sum(1)

    def zq_x(v):
        v = (v - min_proj) / (max_proj - min_proj) * q
        v = np.floor(v).astype(np.int64) % q
        if scale:
            v = v * scale
            v = np.floor(v).astype(np.int64) % q
        return v

    D = np.stack([zq_x(P @ np.load(p1).flatten()) - zq_x(P @ np.load(p2).flatten())
                  for p1, p2 in pairs]).astype(np.int64)
    b = D.astype(np.float64).copy()                                  # Babai nearest-plane, batched
    for k in reversed(range(Bz.shape[0])):
        b = b - np.outer(np.rint((b @ G[k]) / Gn[k]), Bz[k])
    v = np.rint(D - b).astype(np.int64)                              # nearest lattice point
    return (v % q == 0).all(axis=1)                                  # success <=> codeword 0


def verify(scheme, dataset, model, pairs, id_of):
    if scheme == "unprotected":
        return verify_unprotected(pairs, id_of)
    if scheme in ("facialfe_leech", "facialfe_leech_randomised"):
        return verify_leech(scheme, pairs, id_of)
    if scheme == "ourfe":
        return verify_ourfe(scheme, pairs, id_of)
    if scheme == "facialfe_e8":
        return verify_e8(model, pairs, id_of)
    if scheme == "randomfe":
        return verify_randomfe(dataset, model, pairs, id_of)
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["unprotected", *PROTECTED])
    ap.add_argument("--dataset", default="celeba")
    ap.add_argument("--model", default="facenet")
    ap.add_argument("--mode", default="train", choices=["test", "train"],
                    help="test -> test+test2 pairs; train -> train+test_seen pairs")
    ap.add_argument("--samples", type=int, default=2000, help="0 = all genuine/impostor pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(HERE, "tpr_fpr_results.csv"))
    args = ap.parse_args()

    id_of = id_lookup(args.dataset)
    same, diff = build_pairs(args.dataset, args.model, mode=args.mode,
                             samples=args.samples, seed=args.seed)
    tp = verify(args.scheme, args.dataset, args.model, same, id_of)
    fp = verify(args.scheme, args.dataset, args.model, diff, id_of)
    tpr, fpr = float(tp.mean()), float(fp.mean())
    print(f"[{args.scheme}] {args.dataset}/{args.model} {args.mode}: "
          f"TPR={tpr:.4f} (n_genuine={len(tp)})  FPR={fpr:.4f} (n_impostor={len(fp)})")

    row = {"dataset": args.dataset, "model": args.model, "mode": args.mode, "scheme": args.scheme,
           "n_genuine": len(tp), "n_impostor": len(fp), "tpr": round(tpr, 4), "fpr": round(fpr, 4)}
    cols = list(row.keys()); key = ["dataset", "model", "mode", "scheme"]
    new = pd.DataFrame([row], columns=cols)
    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        df = pd.concat([df[~(df[key] == pd.Series({k: row[k] for k in key})).all(axis=1)], new],
                       ignore_index=True)
    else:
        df = new
    df.sort_values(key).to_csv(args.csv, index=False)
    print(f"[csv] {len(df)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
