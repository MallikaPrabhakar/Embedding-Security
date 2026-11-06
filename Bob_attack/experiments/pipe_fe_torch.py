"""Differentiable (straight-through) torch equivalents of pipe_fe_transforms.py.

WHY: the numpy FE in pipe_fe_transforms.py is non-differentiable (rint quantization + lattice
decode = zero gradient a.e.), so GMI/DMI cannot backprop the identity loss to z for FE schemes.
These torch versions use a STRAIGHT-THROUGH ESTIMATOR:

    out = exact.detach() + (surrogate - surrogate.detach())

so the FORWARD value is bit-identical to the numpy FE (the trained classifier sees the exact
protected helper -- no train/attack mismatch), while the BACKWARD pass uses a smooth surrogate
whose gradient is the analytically-correct one almost everywhere:

  * leech secure sketch  s = (c - w_int) % Q ,  w_int = scale*normalize(PROJ@emb)+scale
        d(out)/d(emb) = -(scale/(Q/2)) * d(normalize(PROJ@emb))/d(emb)
        (decode(c) is locally constant -> 0 grad; mod has grad 1; _to_real is linear)
  * facialfe_e8  s = decode(emb/t) - emb/t   ->  d(out)/d(emb) = -1/t

The surrogate is a coarse, per-Voronoi-cell-local approximation (discontinuous at cell
boundaries), so the resulting attack is weaker than on unprotected embeddings -- by design.
"""
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'PIPE_attack', 'datagen'))
import leech_helper as _L          # noqa: E402
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pipe_fe_transforms as _np_fe  # noqa: E402

Q = _L.Q
_PROJ_NP = _L.PROJECTION                       # (504, 512)
LEECH_SCALE = _np_fe.FACIALFE_LEECH_SCALE
E8_T = _np_fe.FACIALFE_E8_T


def _ste(exact, surrogate):
    """forward = exact value, backward = surrogate gradient."""
    return exact.detach() + (surrogate - surrogate.detach())


def _exact_batch(np_fe, emb, names):
    """Run the bit-exact numpy FE per row; return a torch tensor matching emb's device/dtype."""
    e = emb.detach().cpu().numpy().astype(np.float64)
    nm = names if names is not None else [None] * e.shape[0]
    out = np.stack([np.asarray(np_fe(e[i], nm[i]), dtype=np.float64) for i in range(e.shape[0])])
    return torch.from_numpy(out).to(device=emb.device, dtype=emb.dtype)


def get_torch_transform(scheme, dataset='celeba', centered=True, per_image_seed=False):
    """Return f(emb[bs,512], names) -> torch[bs, fe_dim], differentiable via STE.

    per_image_seed is forwarded to the exact numpy FE (the forward), so the attack's codewords
    match however the classifier was trained (per-identity vs per-image)."""
    np_fe = _np_fe.get_transform(scheme, dataset=dataset, centered=centered, per_image_seed=per_image_seed)

    if scheme == 'unprotected':
        return lambda emb, names=None: emb  # already differentiable

    if scheme in ('facialfe_leech', 'facialfe_leech_randomised'):
        def f(emb, names=None):
            proj = torch.as_tensor(_PROJ_NP, device=emb.device, dtype=emb.dtype)   # (504,512)
            w = emb @ proj.t()                                                     # (bs,504)
            w = w / (w.norm(dim=1, keepdim=True) + 1e-12)
            surrogate = -(LEECH_SCALE / (Q / 2.0)) * w
            return _ste(_exact_batch(np_fe, emb, names), surrogate)
        return f

    if scheme == 'facialfe_e8':
        def f(emb, names=None):
            surrogate = -(1.0 / E8_T) * emb                                        # (bs,512)
            return _ste(_exact_batch(np_fe, emb, names), surrogate)
        return f

    if scheme == 'ourfe':
        raise NotImplementedError(
            "ourfe (our defense) is decode-first AND its public helper is a one-way keyed hash, "
            "so there is no differentiable surrogate -> non-diff numpy FE under no_grad.")

    raise ValueError(f"unknown / unsupported scheme: {scheme}")
