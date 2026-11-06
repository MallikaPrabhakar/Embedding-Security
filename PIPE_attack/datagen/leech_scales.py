# Determine FE params (quantisation scale) for the leech-lattice schemes.
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from leech_helper import _preprocess, _to_int, _decode_vector_int
import datagen.datagen_facialfe_leech as datagen_facialfe_leech               # c = decode(w_int)
import datagen.datagen_facialfe_leech_randomised as datagen_facialfe_leech_randomised    # c = random codeword
import entropy_wrapper_facialfe_e8          # ss_gen/ss_rec wrapper over the E8 decoder (pure numpy)
from helper import build_pairs
Q = 1024
DIM = 24
N_BLOCKS = 21

# Each scheme maps to the per-block recovery criterion its gen/rep implements.
# The fast sweep evaluates this criterion directly (decodes are cached per
# embedding); the slow gen/rep path (GENREP, --exact) is the authoritative
# reference and the two are verified to agree per block.
#   'decodefirst' -> match iff decode(w_int) == decode(wp_int)   (codewords agree)
#   'shift'       -> match iff decode((wp_int - w_int) % Q) == 0 (probe in enrol cell)
SCHEMES = {
    'facialfe':            'shift',        # baseline: c = decode(w_int)
    'facialfe_randomised': 'shift',        # randomised c (the random codeword cancels)
}

GENREP = {
    'facialfe':            (datagen_facialfe_leech.ss_gen,            datagen_facialfe_leech.ss_rec),
    'facialfe_randomised': (datagen_facialfe_leech_randomised.ss_gen, datagen_facialfe_leech_randomised.ss_rec),
    'facialfe_e8':         (entropy_wrapper_facialfe_e8.ss_gen,           entropy_wrapper_facialfe_e8.ss_rec),
}

def parse_args():
    parser = argparse.ArgumentParser(description='Leech-FE scale AUROC sweep across schemes')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--schemes', nargs='+', default=list(SCHEMES),
                        choices=list(SCHEMES), help='which schemes to sweep')
    parser.add_argument('--samples', type=int, default=2000,
                        help='total same+diff pairs to evaluate')
    parser.add_argument('--seed', type=int, default=0)
    # Scale grid, expressed in units of Q//8 (the integer step of the decoder).
    parser.add_argument('--scale-start', type=float, default=11)
    parser.add_argument('--scale-stop', type=float, default=11.5)
    parser.add_argument('--scale-step', type=float, default=0.01)
    parser.add_argument('--exact', action='store_true',
                        help='use the authoritative (slow) gen/rep path instead of the cached criterion')
    return parser.parse_args()



def make_loader():
    """Return a loader that reads each embedding .npy from disk only once."""
    cache = {}

    def load_emb(path):
        if path not in cache:
            cache[path] = np.load(path)
        return cache[path]

    return load_emb


def make_preprocessor(load_emb):
    """Return a cached _preprocess(w).

    _preprocess (random projection + unit-norm) is scale-independent, so it
    only needs to run once per embedding and can be reused across every scale."""
    cache = {}

    def preprocess(path):
        if path not in cache:
            cache[path] = _preprocess(load_emb(path))
        return cache[path]

    return preprocess


def abs_dist(x, y):
    return np.linalg.norm(x.astype(np.float32) - y.astype(np.float32))


def count_matching_blocks(x, y, dim=DIM):
    """Number of length-`dim` blocks that are identical between x and y."""
    return sum(abs_dist(x[i*dim:(i+1)*dim], y[i*dim:(i+1)*dim]) < 1e-6
               for i in range(len(x) // dim))


# --- Per-scheme recovery criteria (fast path) ------------------------------
# Each criterion is (embed_fn, pair_fn):
#   embed_fn(preprocessed_w, scale) -> a per-embedding array, cached once per
#       (file, scale) so each unique embedding is processed once.
#   pair_fn(embed_a, embed_b)       -> matching-block count for that pair.
_ZERO = np.zeros(N_BLOCKS * DIM, dtype=np.uint32)


def _embed_decodefirst(prep, scale):
    """Decoded codeword of the embedding (the expensive decode is cached here)."""
    return _decode_vector_int(_to_int(prep, scale=scale))[0] % Q


def _pair_decodefirst(ca, cb):
    return count_matching_blocks(ca, cb)


def _embed_shift(prep, scale):
    """Just the quantised integer embedding; the decode happens per pair."""
    return _to_int(prep, scale=scale).astype(np.int64)


def _pair_shift(wa, wb):
    # match iff the probe lies in the enrolment block's cell: decode(wp-w)==0.
    diff = ((wb - wa) % Q).astype(np.uint32)
    dz = _decode_vector_int(diff)[0] % Q
    return count_matching_blocks(dz, _ZERO)


CRITERIA = {
    'decodefirst': (_embed_decodefirst, _pair_decodefirst),
    'shift':       (_embed_shift,       _pair_shift),
}


def embed_cache(files, preprocess, embed_fn, scale):
    """Apply embed_fn to each *unique* file once at this scale."""
    return {f: embed_fn(preprocess(f), scale) for f in tqdm(files, desc='embed', leave=False)}


def pair_match_blocks(ss_gen, ss_rec, load_emb, w, wp, scale):
    """Authoritative (slow) per-pair score via a scheme's gen/rep (--exact)."""
    s, _c = ss_gen(load_emb(w), scale=scale)
    ref = ss_rec(load_emb(w), s, scale=scale)       # genuine recovery (always succeeds)
    out = ss_rec(load_emb(wp), s, scale=scale)      # probe recovery
    return count_matching_blocks(ref, out)


def best_auroc_over_blocks(same_counts, diff_counts, scale, scheme_name, blocks):
    """One record per block threshold; return (records, best_auroc).

    The decision `count >= b` is binary, so its ROC has a single operating
    point (fpr, tpr) and the AUROC has the closed form 0.5*(tpr + (1 - fpr)) =
    (TPR + TNR)/2. This is exactly what roc_auc_score returns for a binary
    score (verified), without the per-block sklearn calls. The constant-score
    case (all-match / all-miss) falls out naturally as 0.5."""
    records, best = [], 0.5
    for b in blocks:
        tpr = float(np.mean(same_counts >= b)) if len(same_counts) else 0.0
        fpr = float(np.mean(diff_counts >= b)) if len(diff_counts) else 0.0
        auroc = 0.5 * (tpr + (1.0 - fpr))
        best = max(best, auroc)
        records.append({'scheme': scheme_name, 'block': int(b), 'scale': scale,
                        'auroc': auroc, 'fpr': fpr, 'tpr': tpr})
    return records, best


def scheme_counts(scheme_name, same_pairs, diff_pairs, unique_files, scale,
                  load_emb, preprocess, exact):
    """Per-pair matching-block counts for one scale, picking the scheme's path.

    Fast path: cache the scheme's per-embedding criterion array once per unique
    file, then score every pair from the cache. Exact path: run the
    scheme's authoritative gen/rep for every pair."""
    if exact:
        ss_gen, ss_rec = GENREP[scheme_name]
        score = lambda a, b: pair_match_blocks(ss_gen, ss_rec, load_emb, a, b, scale)
    else:
        embed_fn, pair_fn = CRITERIA[SCHEMES[scheme_name]]
        cache = embed_cache(unique_files, preprocess, embed_fn, scale)
        score = lambda a, b: pair_fn(cache[a], cache[b])
    same = np.array([score(a, b) for a, b in same_pairs], dtype=np.int32)
    diff = np.array([score(a, b) for a, b in diff_pairs], dtype=np.int32)
    return same, diff


def sweep_scheme(scheme_name, same_pairs, diff_pairs, scales, model, load_emb, preprocess, exact=False):
    """Run the full scale sweep for one scheme, writing/ resuming a per-scheme CSV."""
    blocks = list(range(1, N_BLOCKS + 1))
    csv_path = f'leech_scales/{model}_{scheme_name}_auroc.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Each embedding file is processed once per scale; collect the set.
    unique_files = sorted({f for pair in same_pairs + diff_pairs for f in pair})

    # resume from prior run if present
    if os.path.exists(csv_path):
        prior = pd.read_csv(csv_path)
        records = prior.to_dict('records')
        done_scales = {float(s) for s in prior['scale'].unique()}
        print(f'[{scheme_name}] resuming with {len(done_scales)} scales already done')
    else:
        records, done_scales = [], set()

    scale_bar = tqdm([s for s in scales if s not in done_scales],
                     desc=f'{model}/{scheme_name} scales')
    for scale in scale_bar:
        scale_bar.set_postfix(scale=f'{scale:.2f}')
        same_counts, diff_counts = scheme_counts(scheme_name, same_pairs, diff_pairs,
                                                 unique_files, scale, load_emb, preprocess, exact)

        new_records, best = best_auroc_over_blocks(same_counts, diff_counts, scale, scheme_name, blocks)
        records.extend(new_records)
        scale_bar.set_postfix(scale=f'{scale:.2f}', best_auroc=f'{best:.3f}')

        done_scales.add(scale)
        pd.DataFrame(records, columns=['scheme', 'block', 'scale', 'auroc', 'fpr', 'tpr']).to_csv(csv_path, index=False)
    return csv_path


def main():
    args = parse_args()
    scales = [float(s) for s in
              np.arange(args.scale_start, args.scale_stop, args.scale_step) * float(Q // 8)]

    same_pairs, diff_pairs = build_pairs(args.dataset, args.model, mode='train', samples=args.samples, seed=args.seed)
    print(f'{len(same_pairs)} same pairs, {len(diff_pairs)} diff pairs, '
          f'{len(scales)} scales, schemes={args.schemes}, exact={args.exact}')

    load_emb = make_loader()                  # disk read cached, shared across schemes
    preprocess = make_preprocessor(load_emb)  # + _preprocess cached
    for scheme_name in args.schemes:
        out = sweep_scheme(scheme_name, same_pairs, diff_pairs, scales, args.model,
                           load_emb, preprocess, exact=args.exact)
        print(f'[{scheme_name}] wrote {out}')


if __name__ == '__main__':
    main()
