"""
Protected-scheme attack evaluation (fuzzy-extractor key recovery). Handles both the
leech-lattice schemes (facialfe_leech / facialfe_leech_randomised / ourfe) and the
E8-lattice scheme (facialfe_e8).

Unlike scripts/evaluation.py (which measures face-matching closeness of the reconstructed
image to the original via Facenet512 distance), this script measures whether the attacker
recovers the *protected codeword* (the FE secret):

  1. Embed each of the N reconstructed sample images (from the diffusion model) with DeepFace
     Facenet512 (normalization=ArcFace, detector_backend=skip) -> w'  (512-d) -- the SAME
     extractor that produced the enrolment embeddings.
  2. Run the scheme's recovery with the stored per-sample sketch s:
        facialfe_leech / facialfe_leech_randomised:  cp = decode_leech((to_int(w') + s) mod Q)
        ourfe (decode-first keyed hash):             recompute keyed hash of decode(to_int(w'))
        facialfe_e8:                                  cp = decode_E8(w'/t + s)
  3. Compare the recovered codeword cp to the enrolled codeword c, per lattice block
     (24-d Leech -> 21 blocks; 8-d E8 -> 64 blocks). The leech-family schemes load c from a
     stored codewords dir; for facialfe_e8 the enrolled key is reconstructed as c = s + w_enroll
     (w_enroll = unprotected_embedding / t), so no codewords dir is needed.

ASR (attack success rate) = fraction of samples whose recovered codeword matches the enrolled
codeword in >= --block_threshold blocks (default: per-scheme; full key for facialfe, ourfe=4).

Runs on CPU in the `deepface` conda env (Facenet512 is CPU there). Configurable scheme:
  --scheme {facialfe_leech, facialfe_leech_randomised, ourfe, facialfe_e8}
"""
import os
import sys
import glob
import argparse
import time

import numpy as np
import cv2
from deepface import DeepFace

# leech_helper / datagen_facialfe_e8 live in PIPE_attack/datagen (sibling of this scripts/ dir)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datagen"))
from leech_helper import DIM, Q, _preprocess, _to_int, _decode_vector_int, _vector_hash  # noqa: E402

E8_DIM = 8  # facialfe_e8: 512-d embedding -> 64 blocks of 8

# per-scheme leech scale (FACENET_SCALE in the datagen scripts) / e8 facenet invscale t; also
# names the dataset dir: facenet_scale<scale> for the leech family, facenet_invscale<t> for e8.
SCHEME_SCALE = {
    "facialfe_leech": 1461.76,
    "facialfe_leech_randomised": 1461.76,
    "ourfe": 144.64,
    "facialfe_e8": 7.25,
}

# lattice block dim and #blocks per scheme
SCHEME_DIM = {
    "facialfe_leech": DIM,             # 24
    "facialfe_leech_randomised": DIM,
    "ourfe": DIM,
    "facialfe_e8": E8_DIM,             # 8
}
SCHEME_NBLOCKS = {
    "facialfe_leech": 504 // DIM,      # 21
    "facialfe_leech_randomised": 504 // DIM,
    "ourfe": 504 // DIM,               # 21
    "facialfe_e8": 512 // E8_DIM,      # 64
}

# match criterion (#blocks that must agree) per scheme. The facialfe variants require the full
# codeword (21 leech / 64 e8). ourfe (our defense, keyed-hash sketch) uses the >= 4 operating point.
SCHEME_BLOCK_THRESHOLD = {
    "facialfe_leech": 504 // DIM,             # 21
    "facialfe_leech_randomised": 504 // DIM,  # 21
    "ourfe": 4,
    "facialfe_e8": 512 // E8_DIM,             # 64
}

# output-file tag (preserves the existing test_protected_blocks_<tag>.npy names; e8 stays "e8")
SCHEME_TAG = {
    "facialfe_leech": "facialfe_leech",
    "facialfe_leech_randomised": "facialfe_leech_randomised",
    "ourfe": "ourfe",
    "facialfe_e8": "e8",
}


def ss_rec(scheme, wp, s, scale):
    """Recover codeword from a probe embedding wp and stored sketch s (leech-family / ourfe).

    For ourfe the sketch is a one-way keyed hash, so we cannot invert it; instead s
    is the stored per-block keys and we RECOMPUTE the keyed hash of the probe's decoded
    codeword. The recomputed hashes are then compared per block to the stored hashes."""
    wp_int = _to_int(_preprocess(wp), scale=scale)
    if scheme == "ourfe":  # decode-first, then recompute the keyed hash with the public keys s
        wp_c = _decode_vector_int(wp_int)[0]
        return _vector_hash(wp_c, s)
    # facialfe_leech / facialfe_leech_randomised: add sketch then decode
    cp = (wp_int + s) % Q
    return _decode_vector_int(cp)[0]


def count_matching_blocks(x, y, dim=DIM):
    return int(sum(
        np.linalg.norm(x[i * dim:(i + 1) * dim].astype(np.float64)
                       - y[i * dim:(i + 1) * dim].astype(np.float64)) < 1e-6
        for i in range(len(x) // dim)
    ))


def count_matching(scheme, cp, c, dim=DIM):
    """Number of agreeing lattice blocks, or for ourfe the number of agreeing
    per-block keyed hashes (cp and c are 21-element hash vectors)."""
    if scheme == "ourfe":
        return int(np.sum(np.asarray(cp) == np.asarray(c)))
    return count_matching_blocks(cp, c, dim=dim)


def embed_images(images, normalization="ArcFace"):
    """Facenet512 embedding of each RGB uint8 image (same call as enrolment)."""
    embs = []
    for img in images:
        rep = DeepFace.represent(img, model_name="Facenet512", enforce_detection=False,
                                 detector_backend="skip", normalization=normalization)
        embs.append(np.asarray(rep[0]["embedding"], dtype=np.float64))
    return embs


def main():
    args = create_argparser().parse_args()
    os.environ["DEVICE_ID"] = str(args.device_id)
    scheme = args.scheme
    scale = SCHEME_SCALE[scheme]
    dim = SCHEME_DIM[scheme]
    n_blocks = SCHEME_NBLOCKS[scheme]
    bt = args.block_threshold if args.block_threshold is not None else SCHEME_BLOCK_THRESHOLD[scheme]
    is_e8 = scheme == "facialfe_e8"

    result_path = os.path.join(args.output_dir, args.result_file)

    if is_e8:
        # E8: sketches live in the protected embeddings dir; the enrolled key is reconstructed as
        # s + w_enroll (w_enroll = unprotected_embedding / t), so no codewords dir is needed.
        import datagen_facialfe_e8 as e8
        prot = os.path.join(args.datasets_root, scheme, args.dataset,
                            f"facenet_invscale{scale}", args.eval_dset, "embeddings")
        unprot = os.path.join(args.datasets_root, "unprotected", args.dataset, "facenet",
                              args.eval_dset, "embeddings")
        sk_files = sorted(glob.glob(os.path.join(prot, "*.npy")))
        cw_files = []
    else:
        # leech-family / ourfe: per-sample sketch + stored codeword in dedicated dirs
        prot = os.path.join(args.datasets_root, scheme, args.dataset,
                            f"facenet_scale{scale}", args.eval_dset)
        sk_files = sorted(glob.glob(os.path.join(prot, "sketches", "*.npy")))
        cw_files = sorted(glob.glob(os.path.join(prot, "codewords", "*.npy")))

    print(f"[eval_protected] scheme={scheme} scale={scale} split={args.eval_dset}")
    print(f"  reconstructions: {result_path}")
    print(f"  sketches: {prot}")
    recon = np.load(result_path)["arr_0"]  # (N, 64, 64, 3) uint8 RGB
    n = len(recon) if args.num <= 0 else min(args.num, len(recon))
    assert len(sk_files) >= n, "fewer sketches than samples"
    if not is_e8:
        assert len(cw_files) >= n, "fewer codewords than samples"
        cw_files = cw_files[:n]
    recon = recon[:n]; sk_files = sk_files[:n]

    start = time.time()
    embs = embed_images(recon)
    print(f"  embedded {n} reconstructions in {(time.time()-start)/60:.2f} min")

    blocks = np.empty(n, dtype=np.int32)
    for i in range(n):
        s = np.load(sk_files[i])
        if is_e8:
            w_enroll = np.load(os.path.join(unprot, os.path.basename(sk_files[i]))).flatten() / scale
            c_enroll = s + w_enroll                       # = decode_E8(w_enroll)
            c_rec = e8.decode(embs[i] / scale + s)        # recovered key from the probe
            blocks[i] = count_matching_blocks(c_rec, c_enroll, dim=dim)
        else:
            c = np.load(cw_files[i])
            cp = ss_rec(scheme, embs[i], s, scale)
            blocks[i] = count_matching(scheme, cp, c, dim=dim)

    asr = float(np.mean(blocks >= bt))
    np.save(os.path.join(args.output_dir, f"{args.eval_dset}_protected_blocks_{SCHEME_TAG[scheme]}.npy"), blocks)
    print(f"  blocks matched (of {n_blocks}): mean={blocks.mean():.2f} median={int(np.median(blocks))} "
          f"max={blocks.max()} (#>= {bt}={int(np.sum(blocks>=bt))})")
    if is_e8:  # e8: near-full threshold breakdown (as the former eval_protected_e8.py)
        for thr in (n_blocks, n_blocks - 1, n_blocks - 2, 60, 56):
            if thr <= n_blocks:
                print(f"  ASR(>= {thr}/{n_blocks}): {float(np.mean(blocks>=thr)):.4f}")
    print(f"ASR (>= {bt}/{n_blocks} blocks): {asr:.4f}")

    # optional sanity: re-embed original images, run genuine recovery (expect ~1.0). Uses ss_rec +
    # stored codewords, so leech-family / ourfe only (e8 has no codewords dir).
    if args.sanity_n > 0 and not is_e8:
        orig_dir = os.path.join(args.original_dir, args.eval_dset, "images")
        img_files = sorted(glob.glob(os.path.join(orig_dir, "*")))[:args.sanity_n]
        oimgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in img_files]
        oembs = embed_images(oimgs)
        gb = np.array([count_matching(scheme, ss_rec(scheme, oembs[i], np.load(sk_files[i]), scale),
                                       np.load(cw_files[i]), dim=dim) for i in range(len(oimgs))])
        print(f"  [sanity] genuine recovery on {len(oimgs)} originals: "
              f"mean_blocks={gb.mean():.2f} ASR(full)={float(np.mean(gb==n_blocks)):.4f}")
    elif args.sanity_n > 0 and is_e8:
        print("  [sanity] not supported for facialfe_e8 (no codewords dir); skipping")


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", required=True,
                   choices=["facialfe_leech", "facialfe_leech_randomised", "ourfe", "facialfe_e8"])
    p.add_argument("--output_dir", required=True,
                   help="results dir holding the reconstructed-samples npz")
    p.add_argument("--result_file", default="test_samples_1000x64x64x3.npz")
    p.add_argument("--eval_dset", default="test")
    p.add_argument("--dataset", default="celeba",
                   help="which dataset's protected sketches/codewords to use")
    p.add_argument("--datasets_root", default="datasets",
                   help="root holding <scheme>/<dataset>/<scale_dir>/<split>/{sketches,codewords} "
                        "(leech family) or .../embeddings plus unprotected/<dataset>/facenet/... (e8)")
    p.add_argument("--original_dir", default="datasets/celeba",
                   help="for --sanity_n: <original_dir>/<eval_dset>/images (leech family / ourfe)")
    p.add_argument("--block_threshold", type=int, default=None,
                   help="blocks needed to count a success (default: per-scheme; ourfe=4, "
                        "facialfe_leech=21, facialfe_e8=64)")
    p.add_argument("--num", type=int, default=0, help="limit #samples (0 = all)")
    p.add_argument("--sanity_n", type=int, default=0,
                   help="re-embed this many originals to verify genuine recovery (~1.0); "
                        "leech family / ourfe only")
    p.add_argument("--device_id", default="0")
    return p


if __name__ == "__main__":
    main()
