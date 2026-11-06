# Ports of the PIPE_attack/datagen fuzzy-extractor procedures, operating on a
# facenet *last_bn* (unnormalised, 512-d) embedding -- the exact vector PIPE stores
# under datasets/unprotected/<dataset>/facenet.
#
# IMPORTANT -- what these return (Bob attack convention, NOT the PIPE-stored embedding):
#   Bob is given the FULL PUBLIC HELPER `p` of each scheme -- everything an attacker
#   actually receives -- *ignoring* the secret codeword and any hash key. We deliberately
#   do NOT strip the codeword by decoding (PIPE's stored "embeddings" do that, e.g.
#   facialfe_leech_randomised stores the mod-Lambda residual). The full helper can be
#   larger than PIPE's stored embedding -- that is fine.
#
#   unprotected               -> w                       the 512-d last_bn embedding unchanged
#   facialfe_e8               -> s = decode(w)-w          E8 secure sketch (public), w = e/t           (512)
#   facialfe_leech            -> _to_real(s),  s=(c-w)%Q  Leech secure sketch (public)                 (504)
#   facialfe_leech_randomised -> _to_real(s),  s=(c-w)%Q  Leech sketch, random per-id codeword c        (504)
#                                                          (the PUBLIC sketch, NOT the codeword-stripped residual)
#   ourfe                     -> [keys, hashes]           decode-first per-block keyed-hash helper       (42)
#
# leech codewords use numpy np.random.seed(id) (reproducible here) so all images of a
# person share the per-identity secret codeword, exactly like PIPE datagen.
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # Bob_attack/experiments
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))  # repo root (Embedding-Security)

# leech_helper builds/loads the leech-decoding C library on import (needs gcc once).
sys.path.insert(0, os.path.join(_REPO, 'PIPE_attack', 'datagen'))
import leech_helper as _L  # noqa: E402

Q = _L.Q

# ---- facenet scales, mirroring the PIPE datagen scripts ----
FACIALFE_E8_T = 7.25          # datagen_facialfe_e8.py  (ts = [7.25] for facenet)
FACIALFE_LEECH_SCALE = 1461.76  # datagen_facialfe_leech.py / _randomised.py
OURFE_SCALE = 144.64          # datagen_ourdef.py (our defense, decode-first keyed-hash sketch)


# =========================== id map (for seeded schemes) ===========================
_ID_MAPS = {}


def _id_of(img_name, dataset='celeba'):
    """Map an image file name (e.g. '000001.jpg') to its integer CelebA identity.

    For celeba we use the COMPLETE map GAN_attacks/attack_dataset/CelebA/identity_CelebA.txt
    (a verified id-consistent superset of PIPE_attack/datasets/celeba/img_id_map.csv: 12177/12177
    overlap, 0 mismatches) so the seeded FE schemes don't KeyError on training images that
    img_id_map.csv doesn't cover. Other datasets fall back to img_id_map.csv."""
    if dataset not in _ID_MAPS:
        if dataset == 'celeba':
            path = os.path.join(_REPO, 'GAN_attacks', 'attack_dataset', 'CelebA', 'identity_CelebA.txt')
            m = {}
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        m[parts[0]] = int(parts[1])
            _ID_MAPS[dataset] = m
        else:
            import pandas as pd
            df = pd.read_csv(os.path.join(_REPO, 'PIPE_attack', 'datasets', dataset, 'img_id_map.csv'))
            _ID_MAPS[dataset] = dict(zip(df['img'], df['id']))
    return int(_ID_MAPS[dataset][img_name])


# =============================== E8 (facialfe_e8) ==================================
def _decode_d8(x):
    x_ = np.round(x)
    if np.sum(x_) % 2 == 0:
        return x_
    i = np.argmin(np.abs(x - x_))
    ei = np.zeros_like(x_)
    ei[i] = 1
    return x_ + ei if x_[i] > x[i] else x_ - ei


def _decode_e8(x):
    x1 = _decode_d8(x)
    x2 = _decode_d8(x - 0.5) + 0.5      # NB: + 0.5 (the second E8 coset), matching the numpy datagen
    return x1 if np.linalg.norm(x - x1) < np.linalg.norm(x - x2) else x2


def _decode_e8_vec(x):
    m = len(x)
    val = np.zeros_like(x)
    for k in range(m // 8):
        val[k * 8:(k + 1) * 8] = _decode_e8(x[k * 8:(k + 1) * 8])
    return val


def facialfe_e8(emb, img_name=None, t=FACIALFE_E8_T):
    w = np.asarray(emb, dtype=np.float64) / t
    c = _decode_e8_vec(w)
    return (c - w).astype(np.float64)              # public secure sketch s, shape (512,)


# ===================== Leech schemes (facialfe_leech*) ====================
# Both return the PUBLIC sketch s mapped to real. By default raw _to_real(s) (no
# centering); with centered=True the sketch is mod-Q centered first, _to_real((s+Q//2)%Q),
# which gives a single bump instead of the wrap-around U-shape. In BOTH cases this is the
# public sketch s (codeword NOT stripped) -- centered != the codeword-stripped residual.
# s lives in [0, Q) so _to_real(s) = (s - Q//2)/(Q//2) in [-1, 1).
def _sketch_to_real(s, centered):
    return _L._to_real((s + Q // 2) % Q) if centered else _L._to_real(s)


# Per-identity Leech codeword cache. The secret codeword depends ONLY on the identity
# (np.random.seed(_id_of(name)) -> codeword), so all images of a person share it. Generating a
# 21-block Leech codeword is expensive; cache it by identity so it's built once per person
# instead of once per image. Bit-exact: seed(iden)->codeword is deterministic. Keyed by the
# IDENTITY (_id_of value), NOT the image name.
_LEECH_CW_CACHE = {}


def _seed_of(img_name, dataset, per_image_seed):
    """Codeword seed. per_image_seed=False (default, backwards-compatible): seed by IDENTITY
    (_id_of) so all of a person's images share the codeword (PIPE datagen behaviour). True:
    seed by the IMAGE NAME so every image gets a distinct codeword -- used by the GAN-attack
    classifier training, where ~27 train + 3 test images per identity span both splits, so a
    per-identity codeword leaks the label (-> trivial 100% classifier)."""
    if per_image_seed:
        return int(os.path.splitext(img_name)[0])   # numeric image name (e.g. '000001.jpg' -> 1)
    return _id_of(img_name, dataset)


def _leech_codeword_for(iden):
    if iden not in _LEECH_CW_CACHE:
        np.random.seed(iden)
        _LEECH_CW_CACHE[iden] = _L.random_leech_codeword_vector_int()
    return _LEECH_CW_CACHE[iden]


def facialfe_leech(emb, img_name=None, scale=FACIALFE_LEECH_SCALE, centered=False):
    w_int = _L._to_int(_L._preprocess(np.asarray(emb, np.float64)), scale=scale)
    c = _L._decode_vector_int(w_int)[0]
    s = (c - w_int) % Q
    return _sketch_to_real(s, centered)            # shape (504,)


def facialfe_leech_randomised(emb, img_name=None, scale=FACIALFE_LEECH_SCALE, dataset='celeba', centered=False, per_image_seed=False):
    w_int = _L._to_int(_L._preprocess(np.asarray(emb, np.float64)), scale=scale)
    if img_name is not None:
        c = _leech_codeword_for(_seed_of(img_name, dataset, per_image_seed))   # cached per seed
    else:
        c = _L.random_leech_codeword_vector_int()            # no name -> non-reproducible (unchanged)
    s = (c - w_int) % Q                            # PUBLIC sketch (codeword NOT stripped)
    return _sketch_to_real(s, centered)            # shape (504,)


# ============================= ourfe (our defense) ===============================
# Decode-first secure sketch whose PUBLIC helper is a per-block keyed BLAKE2b hash. The
# helper Bob receives is the flattened+normalised [keys, hashes] = 42-d vector, built by the
# SAME leech_helper.ourfe_embedding used by PIPE datagen (datagen_ourdef.py), so the two
# pipelines are bit-for-bit identical. The keys are seeded per-identity (np.random.seed(id)),
# exactly like the leech codewords, so the helper is reproducible from an image. The keyed
# hash is one-way -> no clean differentiable surrogate (see pipe_fe_torch.py).
def ourfe(emb, img_name=None, scale=OURFE_SCALE, dataset='celeba', per_image_seed=False):
    w_c = _L._decode_vector_int(_L._to_int(_L._preprocess(np.asarray(emb, np.float64)), scale=scale))[0]
    seed = _seed_of(img_name, dataset, per_image_seed) if img_name is not None else 0
    np.random.seed(seed)
    keys = _L.sample_ourfe_keys(len(w_c) // _L.DIM)
    hashes = _L._vector_hash(w_c, keys)
    return _L.ourfe_embedding(keys, hashes)        # shape (42,)


# ================================== dispatcher ====================================
EMBED_DIMS = {
    'unprotected': 512,
    'facialfe_e8': 512,
    'facialfe_leech': 504,
    'facialfe_leech_randomised': 504,
    'ourfe': _L.ourfe_helper_dim(),  # 42 = flatten [keys, hashes] = 2 * 21 blocks
}

# schemes whose helper depends on the per-identity secret codeword / keys (need img_id)
SEEDED_SCHEMES = {'facialfe_leech_randomised', 'ourfe'}


def get_transform(scheme, dataset='celeba', centered=False, per_image_seed=False):
    """Return a callable f(emb_512_lastbn, img_name=None) -> public helper p (np).

    centered (facialfe_leech / facialfe_leech_randomised): emit the mod-q centered helper --
    _to_real((s+Q//2)%Q) for the leech sketches -- instead of the raw value. Ignored by
    unprotected / facialfe_e8 / ourfe.

    per_image_seed (default False = PIPE per-identity codewords; True = per-image codewords,
    for GAN-attack classifier training -- see _seed_of). Only affects the seeded schemes
    (facialfe_leech_randomised, ourfe)."""
    if scheme == 'unprotected':
        return lambda emb, img_name=None: np.asarray(emb, dtype=np.float64).flatten()
    if scheme == 'facialfe_e8':
        return facialfe_e8
    if scheme == 'facialfe_leech':
        return lambda emb, img_name=None: facialfe_leech(emb, img_name, centered=centered)
    if scheme == 'facialfe_leech_randomised':
        return lambda emb, img_name=None: facialfe_leech_randomised(emb, img_name, dataset=dataset, centered=centered, per_image_seed=per_image_seed)
    if scheme == 'ourfe':
        return lambda emb, img_name=None: ourfe(emb, img_name, dataset=dataset, per_image_seed=per_image_seed)
    raise ValueError(f'unknown PIPE FE scheme: {scheme}')
