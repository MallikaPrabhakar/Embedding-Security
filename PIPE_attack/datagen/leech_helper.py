# %%
# Shared leech-lattice fuzzy-extractor helpers used by the per-scheme datagen
# modules (datagen_facialfe_leech / _randomised, datagen_ourdef) and Bob's
# pipe_fe_transforms. Provides: bindings to the leech-decoding C library
# (Vardy-Be'ery maximum-likelihood decoder, ../../leech-decoding) for blockwise
# 24-dim decoding, real<->Z_Q quantisation, and the per-block keyed-hash sketch.
import ctypes
import hashlib
import os
import subprocess
import sys
import numpy as np

# universal_hash.py is a sibling module (optional sketch backend; see OURFE_HASH_MODE).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import universal_hash  # noqa: E402

# %%
# Locate / build the leech-decoding shared library.
LEECH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'leech-decoding'))
LIB_PATH = os.path.join(LEECH_DIR, 'libleech.so')
if not os.path.exists(LIB_PATH):
    subprocess.check_call([
        'gcc', '-O2', '-fPIC', '-shared',
        os.path.join(LEECH_DIR, 'leech.c'),
        os.path.join(LEECH_DIR, 'leech_utils.c'),
        '-o', LIB_PATH,
    ])

_lib = ctypes.CDLL(LIB_PATH)
_lib.decoder_L24.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
_lib.decoder_L24.restype = None
_lib.decode_pt.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64, ctypes.c_int]
_lib.decode_pt.restype = None

# These mirror the constants in leech-decoding/leech.h.
Q = 1024
# 1.0 in integer leech lattice → SCALE steps in the decoding leech lattice
SCALE = Q // 8
DIM = 24

rng = np.random.default_rng(seed = 3448703)
PROJECTION = rng.standard_normal((504, 512)) / np.sqrt(504) # approximately same norm

def _decode_block_int(t_int: np.ndarray):
    """Closest Leech lattice point to t_int (uint32, shape (24,)).

    Returns (c, d_sq): the decoded codeword c (uint32, shape (24,)) and d_sq, the
    squared Euclidean distance (int) between t_int (the noisy codeword).
    Successful recovery requires d_sq <= 8 * SCALE**2."""
    t_int = np.ascontiguousarray(t_int, dtype=np.uint32)
    cv = ctypes.c_uint64(0)
    d = ctypes.c_uint64(0)
    _lib.decoder_L24(
        t_int.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(cv),
        ctypes.byref(d),
    )
    c = np.zeros(DIM, dtype=np.uint32)
    _lib.decode_pt(c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), cv, 0)
    return c, int(d.value)

def _decode_vector_int(t_int: np.ndarray):
    """Decode a longer vector by decoding each 24-dim block independently."""
    n_blocks = len(t_int) // DIM
    assert len(t_int) % DIM == 0, "input length must be a multiple of 24"
    c = np.zeros_like(t_int, dtype=np.uint32)
    d = np.zeros(n_blocks, dtype=np.uint64)
    for b in range(n_blocks):
        sl = slice(b * DIM, (b + 1) * DIM)
        c[sl], d[b] = _decode_block_int(t_int[sl])
    return c, d


HASH_BYTES = 4  # 32-bit keyed-hash digest per block

# Which keyed hash backs the sketch. 'blake2b' (the default) is the deployed keyed MAC;
# 'poly' / 'multilinear' select a universal hash family instead -- see universal_hash.py
# for what that buys and what it costs. Recovery does NOT consult this: _vector_hash
# infers the family from the stored key's shape, so sketches always verify under the
# family that produced them.
HASH_MODE_BLAKE2B = 'blake2b'
OURFE_HASH_MODE = os.getenv('OURFE_HASH_MODE', HASH_MODE_BLAKE2B)
if OURFE_HASH_MODE != HASH_MODE_BLAKE2B and OURFE_HASH_MODE not in universal_hash.MODES:
    raise ValueError(f'OURFE_HASH_MODE={OURFE_HASH_MODE!r} not in '
                     f'{(HASH_MODE_BLAKE2B, *universal_hash.MODES)}')


def sample_ourfe_keys(n_blocks: int, mode: str = None, rng=None) -> np.ndarray:
    """Per-block sketch keys for our defense, in the layout `_vector_hash` expects.

    blake2b (default) -> shape (n_blocks,), one 32-bit key per block.
    universal families -> shape (n_blocks, key_size), field elements; see universal_hash.

    Draws from the global `np.random` stream by default, so callers keep seeding the
    helper per identity with `np.random.seed(id)` exactly as before."""
    mode = OURFE_HASH_MODE if mode is None else mode
    rng = np.random if rng is None else rng
    if mode == HASH_MODE_BLAKE2B:
        return rng.randint(0, 2 ** 32, size=n_blocks)
    return universal_hash.sample_keys(n_blocks, mode=mode, dim=DIM, rng=rng)


def ourfe_helper_dim(n_blocks: int = 504 // DIM, mode: str = None) -> int:
    """Length of the ourfe public helper -- the PIPE/Bob/GAN conditioning width.

    42 for the default blake2b sketch (21 keys + 21 hashes). A universal-hash key is
    wider than one 32-bit integer, so selecting one widens the helper; use this rather
    than hard-coding a width."""
    mode = OURFE_HASH_MODE if mode is None else mode
    if mode == HASH_MODE_BLAKE2B:
        return 2 * n_blocks
    return n_blocks * (universal_hash.key_size(mode=mode, dim=DIM) + 1)


def _vector_hash(decoded_w: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Blockwise *keyed* hash of a decoded Leech codeword (our defense's sketch).

    A 1-D `keys` selects the default keyed BLAKE2b MAC (below); a 2-D `keys` is a
    universal-hash key and is dispatched to universal_hash.hash_blocks. Either way the
    result is one 32-bit value per block, so the sketch's shape is identical.

    For each 24-dim block b, compute a keyed BLAKE2b MAC over the block's raw
    bytes with the per-block secret key ``keys[b]``:

        h_b = int.from_bytes(blake2b(bytes(w_c[block b]), key=bytes(keys[b]), digest_size=4))

    The digest is 32 bits (digest_size=4), so h_b lives in [0, 2**32) -- it is NOT
    reduced mod Q. Returns an int64 array of one hash per block (shape (n_blocks,)).
    The keyed hash is one-way: it is the PUBLIC sketch of our defense (the codeword
    cannot be inverted from it). Recovery re-derives h'_b from a probe codeword with
    the SAME keys and checks equality per block (see scripts/eval_protected.py).

    blake2b takes a key of <= 64 bytes; we pack the (< 2**32) integer key into 8
    little-endian bytes."""
    assert len(decoded_w) % DIM == 0
    keys = np.asarray(keys)
    if keys.ndim > 1:                       # universal-hash key -> universal_hash family
        return universal_hash.hash_blocks(np.asarray(decoded_w), keys, dim=DIM,
                                          m=universal_hash.HASH_MOD)
    n_blocks = len(decoded_w) // DIM
    blocks = np.ascontiguousarray(decoded_w, dtype=np.uint32)
    hashes = np.empty(n_blocks, dtype=np.int64)
    for b in range(n_blocks):
        sl = slice(b * DIM, (b + 1) * DIM)
        key_bytes = int(keys[b]).to_bytes(8, "little")
        digest = hashlib.blake2b(blocks[sl].tobytes(), key=key_bytes, digest_size=HASH_BYTES).digest()
        hashes[b] = int.from_bytes(digest, "little")
    return hashes


def ourfe_embedding(keys: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    """Public helper of our defense, flattened + scaled to [0, 1) for conditioning.

    Concatenates the per-block keys and the keyed hashes (length 2 * n_blocks = 42
    for the 504-dim / 21-block facenet setting), both 32-bit values in [0, 2**32),
    scaled to [0, 1) by dividing by 2**32 (NOT shifted to [-1, 1)). The [0, 1) range
    is fp16-safe (use_fp16 True). Both the PIPE datagen (datagen_ourdef.py) and Bob's
    pipe_fe_transforms.ourfe build the helper via this single function so the two
    pipelines stay bit-for-bit consistent.

    A universal-hash key (2-D `keys`) is laid out the same way -- each key column, then
    the hashes -- with the field elements scaled by the field modulus instead of 2**32.
    `ourfe_helper_dim` gives the resulting length."""
    keys = np.asarray(keys)
    hashes = np.asarray(hashes, dtype=np.float64)
    if keys.ndim > 1:
        cols = [keys[:, j].astype(np.float64) / np.float64(universal_hash.P_MERSENNE)
                for j in range(keys.shape[1])]
        return np.concatenate(cols + [hashes / np.float64(2 ** 32)])
    v = np.concatenate([keys.astype(np.float64), hashes])
    return v / np.float64(2 ** 32)


def random_leech_codeword_vector_int(blocks=21) -> np.ndarray:
    """block count random Leech lattice point in Z_Q^24."""
    t = np.random.randint(0, Q, size=blocks * DIM).astype(np.uint32)
    for b in range(blocks):
        sl = slice(b * DIM, (b + 1) * DIM)
        t[sl] = _decode_block_int(t[sl])[0]
    return t

def _preprocess(w: np.ndarray) -> np.ndarray:
    """Project embedding to multiple of DIM, then normalize to unit-norm"""
    # Project down to n using a fixed random matrix.
    w = PROJECTION @ w
    w /= np.linalg.norm(w) # re-normalise to unit norm after projection
    return w


def _to_int(w: np.ndarray, scale: float) -> np.ndarray:
    """Quantise a real-valued vector to Z_Q.

    Input convention: `w` is unit-norm (so each component lies in [-1, 1],
    by the inf-norm <= L2-norm bound). The input is unit-normalised here if it
    is not already (e.g. callers that bypass `_preprocess` and pass a raw
    embedding), so the [-1, 1] component bound — and hence the offset/range
    reasoning below — always holds. We shift by +scale before
    rounding so that `(w + 1) * scale` lives in [0, 2*scale] — entirely non-negative.
    The trailing `% Q` ensures values are within [0, Q) even if scale > Q/2.
    The decoder's arithmetic is all mod Q in uint32.

    The +scale offset is invisible to the protocol: it cancels in
    `ss_gen` (s = c - w_int) and `ss_rec` (out = c - s)."""
    w = np.asarray(w, dtype=np.float64)
    norm = np.linalg.norm(w)
    if norm > 0 and not np.isclose(norm, 1.0):
        w = w / norm                      # enforce the unit-norm input convention
    x = np.rint(w * scale + scale)  % Q
    return x.astype(np.uint32)

def _to_real(w_int: np.ndarray) -> np.ndarray:
    """convert the Z_q uint32 vector to [-1, 1]"""
    return (w_int.astype(np.float64) - Q//2) / np.float64(Q//2)