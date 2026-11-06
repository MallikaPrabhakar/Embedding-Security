# Adapter exposing facialfe_e8 through the ss_gen/ss_rec API that entropy_estimate_hard_threshold.py
# (GENREP) expects. datagen_facialfe_e8 decodes the (inverse-)scaled embedding on the
# E8 lattice; its native ss_gen(w,id)/ss_rec(w_,s,id) take no scale and ss_rec returns
# the recovered EMBEDDING. This wrapper:
#   - takes scale = the e8 inverse-scale t; the embedding is divided by t exactly as the
#     e8 datagen does (w = load / t) before decoding;
#   - returns (sketch s = c - w/t, codeword c) from ss_gen, and from ss_rec the recovered
#     secret c = decode(wp/t + s) (the E8 codeword, NOT the embedding), so the entropy
#     histogram is over recovered secrets like the other schemes;
#   - keys codewords as 2*c rounded to int (E8 has half-integer points), hashable for
#     the histogram. Pure numpy (no sage); runs in the ddpm env.
import numpy as np
import datagen_facialfe_e8 as e8


def _key(c):
    return np.rint(np.asarray(c, dtype=np.float64) * 2).astype(np.int64)


def ss_gen(w, scale=None, seed=None):
    ws = np.asarray(w, dtype=np.float64).flatten() / scale   # e8 datagen convention: w = load / t
    c = e8.decode(ws)
    s = c - ws                                               # real-valued sketch (public helper)
    return s, _key(c)


def ss_rec(wp, s, scale=None):
    wps = np.asarray(wp, dtype=np.float64).flatten() / scale
    c = e8.decode(wps + s)                                   # recovered E8 codeword
    return _key(c)
