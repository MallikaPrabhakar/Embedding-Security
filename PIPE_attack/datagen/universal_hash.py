"""Universal (Carter-Wegman) hash families -- an OPTIONAL alternative to the keyed
BLAKE2b MAC that ourfe uses for its per-block sketch by default.

Why this exists
---------------
ourfe's sketch needs a keyed compression function whose per-block collision probability
is small. The deployed default, a keyed BLAKE2b MAC (`leech_helper._vector_hash`), gives
that under a *computational* assumption: BLAKE2b behaves as a PRF. The security theorem
in the paper is stated over a **universal hash family**, where the collision bound is
*information-theoretic* -- it holds for every fixed pair of distinct inputs over the
choice of key, with no assumption on the function. This module supplies such a family so
the implementation can be instantiated exactly as the theorem requires.

Empirically the two agree: the entropy estimates are exactly the same, and the
attack-success rates match to within run-to-run variance.

Selecting a family
------------------
Set `OURFE_HASH_MODE` before importing `leech_helper`:

    OURFE_HASH_MODE=blake2b       # default -- keyed BLAKE2b MAC
    OURFE_HASH_MODE=poly          # polynomial / Carter-Wegman universal hash
    OURFE_HASH_MODE=multilinear   # multilinear universal hash

`leech_helper` infers the family from the *stored key's shape* when recovering, so a
sketch generated under one family can never be verified under another by accident. Note
that a universal key is a pair (or vector) of field elements rather than one 32-bit
integer, so the public helper is correspondingly wider -- see `leech_helper.ourfe_helper_dim`
and pass the value it returns wherever a conditioning width is configured.

The families
------------
Both emit one value in Z_m per block, with m = HASH_MOD = 2**32 to match the 4-byte
BLAKE2b digest, so the sketch's on-disk shape and downstream tooling are unchanged.

  MODE_POLY -- polynomial / Carter-Wegman evaluation hash.
      h_{a,b}(x) = ( b + sum_i x_i * a**(i+1)  mod p )  mod m         key = (a, b)

      For x != y the difference is a nonzero polynomial in `a` of degree <= dim, so it
      vanishes for at most `dim` of the p-1 choices of a:
          eps <= dim/(p-1) + (1 + m/p)/m  ~=  2**-32   (dim=24, p=2**61-1, m=2**32)
      i.e. eps-almost-universal, with a compact 2*61 = 122-bit key.

  MODE_MULTILINEAR -- inner-product / multilinear hash.
      h_{a,b}(x) = ( b + sum_i a_i * x_i  mod p )  mod m    key = (a_0..a_{dim-1}, b)

      For x != y the difference is a nonzero linear form in independent uniform a_i,
      hence uniform on Z_p:  eps = (1 + m/p)/m ~= 2**-32. Strictly (almost-)universal,
      at the cost of a `dim`-element key per block.

All arithmetic is exact Python-int arithmetic (values reach ~2**76 before the single
reduction mod p, which overflows int64), so keys and outputs are stored as int64.
"""
import numpy as np

# A large Mersenne prime; the field the hash is evaluated over.
P_MERSENNE = 2 ** 61 - 1
# Output modulus m. 2**32 matches the blake2b digest_size=4 range, so the sketch stays
# 21 values in [0, 2**32) whichever family is selected.
HASH_MOD = 2 ** 32

MODE_POLY = 'poly'
MODE_MULTILINEAR = 'multilinear'
MODES = (MODE_POLY, MODE_MULTILINEAR)

# Default *universal* family, used when the caller has opted into universal hashing but
# has not named which one. Selecting universal hashing at all is leech_helper's decision
# (OURFE_HASH_MODE, default blake2b) -- this module never reads the environment.
DEFAULT_MODE = MODE_POLY


def _resolve(mode):
    if mode is None:
        return DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f'unknown universal-hash mode {mode!r}, expected one of {MODES}')
    return mode


def key_size(mode=None, dim=24):
    """Number of field elements in one block's key: 2 for poly, dim+1 for multilinear."""
    return 2 if _resolve(mode) == MODE_POLY else dim + 1


def key_mode(keys, dim=24):
    """Infer the family from a stored key array of shape (n_blocks, key_size).

    Recovery uses this, so a sketch generated under one family can never be verified
    under the other by accident."""
    keys = np.atleast_2d(np.asarray(keys))
    k = keys.shape[1]
    if k == dim + 1 and dim != 1:
        return MODE_MULTILINEAR
    if k == 2:
        return MODE_POLY
    raise ValueError(
        f'key array has {k} columns per block; expected 2 (poly) or {dim + 1} '
        f'(multilinear, dim={dim}).')


def sample_keys(n_blocks, mode=None, dim=24, p=P_MERSENNE, rng=None):
    """Draw uniform per-block keys, shape (n_blocks, key_size(mode, dim)), int64.

    Column layout -- poly: [a, b]; multilinear: [a_0 .. a_{dim-1}, b]. The `a`
    coefficients are drawn from [1, p-1] (a = 0 would make the hash constant in x) and
    `b` from [0, p-1].

    `rng` defaults to the global `np.random` module so callers that seed the legacy
    global stream (`np.random.seed(id)`, as ourfe does to make the helper reproducible
    per identity) keep working unchanged."""
    mode = _resolve(mode)
    rng = np.random if rng is None else rng
    n_a = 1 if mode == MODE_POLY else dim
    a = rng.randint(1, p, size=(n_blocks, n_a), dtype=np.int64)
    b = rng.randint(0, p, size=(n_blocks, 1), dtype=np.int64)
    return np.concatenate([a, b], axis=1)


class UniversalHash:
    """One block's universal hash, keyed by (a, b) over GF(p), reduced into Z_m.

    Constructed either with a fresh random key or an explicit one (for recovery), and
    supports both families; vectors are hashed order-sensitively.

        h = UniversalHash(2**32, dim=24)          # fresh random poly key
        h = UniversalHash(2**32, a=..., b=...)    # explicit key (recovery)
        h(block)                                  # -> int in [0, m)
    """

    def __init__(self, m=HASH_MOD, a=None, b=None, p=P_MERSENNE, mode=None, dim=24, rng=None):
        self.m = int(m)
        self.p = int(p)
        self.dim = int(dim)
        self.mode = _resolve(mode)
        if a is None or b is None:
            key = sample_keys(1, mode=self.mode, dim=self.dim, p=self.p, rng=rng)[0]
            a = key[:-1] if self.mode == MODE_MULTILINEAR else key[0]
            b = key[-1]
        if self.mode == MODE_MULTILINEAR:
            self.a = [int(v) % self.p for v in np.atleast_1d(a)]
            if len(self.a) != self.dim:
                raise ValueError(f'multilinear key needs dim={self.dim} coefficients, got {len(self.a)}')
        else:
            self.a = int(a) % self.p
            # coefficient of x_i is a**(i+1); precomputed once per key, reused per block.
            self._pw = None
        self.b = int(b) % self.p

    def _powers(self, n):
        """a**1 .. a**n mod p, cached for the longest n seen."""
        if self._pw is None or len(self._pw) < n:
            pw, cur = [], 1
            for _ in range(n):
                cur = (cur * self.a) % self.p
                pw.append(cur)
            self._pw = pw
        return self._pw

    def __call__(self, x):
        """Hash one block (1-D int array / sequence) or a single integer -> int in [0, m)."""
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            xs = [int(x)]
        else:
            xs = np.asarray(x).ravel().tolist()      # -> exact Python ints
        if self.mode == MODE_MULTILINEAR:
            if len(xs) != self.dim:
                raise ValueError(f'multilinear hash keyed for dim={self.dim}, got a length-{len(xs)} block')
            acc = self.b
            for ai, xi in zip(self.a, xs):
                acc += ai * int(xi)
        else:
            pw = self._powers(len(xs))
            acc = self.b
            for i, xi in enumerate(xs):
                acc += int(xi) * pw[i]
        return (acc % self.p) % self.m


def hash_blocks(values, keys, dim=24, m=HASH_MOD, p=P_MERSENNE):
    """Hash a concatenated vector blockwise: one output per `dim`-sized block.

    `values` is the decoded Leech codeword (length n_blocks*dim of Z_Q coordinates);
    `keys` is (n_blocks, key_size) as produced by `sample_keys`. The family is inferred
    from the key shape. Returns int64 hashes of shape (n_blocks,), each in [0, m)."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) % dim:
        raise ValueError(f'values must be 1-D with length a multiple of dim={dim}, got {values.shape}')
    keys = np.atleast_2d(np.asarray(keys))
    n_blocks = len(values) // dim
    if len(keys) != n_blocks:
        raise ValueError(f'expected {n_blocks} block keys, got {len(keys)}')
    mode = key_mode(keys, dim=dim)

    out = np.empty(n_blocks, dtype=np.int64)
    for b in range(n_blocks):
        key = keys[b]
        a = key[:-1] if mode == MODE_MULTILINEAR else key[0]
        h = UniversalHash(m=m, a=a, b=key[-1], p=p, mode=mode, dim=dim)
        out[b] = h(values[b * dim:(b + 1) * dim])
    return out
