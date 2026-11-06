"""Recovered-secret entropy for ourfe (our defense), over its NATIVE keyed-hash
sketch -- the twin of entropy_estimate_hard_threshold.py for datagen_ourdef.ss_gen/ss_rec.

ourfe's deployed sketch (datagen_ourdef) is a per-block keyed BLAKE2b MAC: for a
decoded Leech codeword w_c (N_BLOCKS=21 blocks of 24 dims) it emits ONE 32-bit
hash per block -- an int64 array of length 21, each value in [0, 2**32). So the
recovered object here is 21 blocks of DIM=1 (not 24), and the values are NOT Z_Q
lattice coordinates. Two things therefore differ from entropy_estimate_hard_threshold.py:

  1. NO `% Q` on the recovered/enrolment sketch. Q=1024 would crush a 32-bit hash
     to 10 bits and fabricate block collisions; we histogram the raw hashes.
  2. Blocks are 1-dimensional (DIM=1, N_BLOCKS=21). Per-block equality is EXACT
     integer equality of the hash. We deliberately do NOT use
     leech_scales.count_matching_blocks: it casts blocks to float32, which cannot
     distinguish distinct ~2**32 hashes (float32 has 24 mantissa bits).

Everything else -- population build, the four recovery modes (baseline/elimination
x hard/soft), the greedy soft grouping, the entropy metrics -- mirrors
entropy_estimate_hard_threshold.py. datagen_ourdef.ss_gen/ss_rec already match the (w, scale, id) and
(wp, sketch, scale) call signatures, so no wrapper is needed.
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import datagen_ourdef
from leech_scales import make_loader
# group_codewords is reused verbatim from entropy_estimate_hard_threshold -- it is DIM-agnostic
# except for one reshape, which now takes a `dim` argument (we pass dim=DIM=1).
from entropy_estimate_hard_threshold import (build_population, group_codewords,
                                    h_min, h_shannon, h_collision)

# ourfe's keyed-hash sketch: 21 blocks, ONE 32-bit value each (vs leech's 21x24).
DIM = 1
N_BLOCKS = 21
SCHEME = 'ourfe'


def parse_args():
    parser = argparse.ArgumentParser(description='ourfe keyed-hash recovered-secret entropy estimation')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--unprotected-dir', type=str, default=None,
                        help='base dir of unprotected embeddings')
    parser.add_argument('--splits', nargs='+', default=['train', 'test_seen', 'test', 'test2'],
                        help='img_id_map label splits that form the population')
    parser.add_argument('--samples', type=int, default=0,
                        help='cap the population to this many probe embeddings (0 = all)')
    parser.add_argument('--enrollments', type=int, default=1,
                        help='number of random enrolments to average the entropy over')
    parser.add_argument('--per-enrollment', action='store_true',
                        help='write one row per enrolment (with an enrollment column) '
                             'instead of a single mean-over-enrolments row')
    parser.add_argument('--elimination', action='store_true',
                        help='vote per identity with a strict majority (else per probe)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--blocks-to-match', type=int, default=N_BLOCKS,
                        help=f'soft threshold; hard (exact) when == N_BLOCKS ({N_BLOCKS})')
    parser.add_argument('--scale', type=float, default=datagen_ourdef.FACENET_SCALE,
                        help='leech scale for the decode (default datagen_ourdef.FACENET_SCALE)')
    return parser.parse_args()


def matching_blocks_exact(x, y):
    """Number of identical DIM=1 blocks between two keyed-hash sketches.

    EXACT integer equality -- NOT leech_scales.count_matching_blocks, whose
    float32 cast would merge distinct ~2**32 hashes."""
    return int(np.count_nonzero(np.asarray(x, dtype=np.int64) == np.asarray(y, dtype=np.int64)))


def recover_histogram(ss_gen, ss_rec, df, scale, seed, load_emb, elimination, blocks_to_match):
    """Build one enrolment's keyed-hash sketch, recover it across the population,
    and return (counter, stats). Mirrors entropy_estimate_hard_threshold.recover_histogram, but the
    recovered object is the RAW hash sketch (NO % Q) and blocks are DIM=1, so
    block equality is exact integer equality (matching_blocks_exact)."""
    rng = np.random.default_rng(seed)
    w0_idx = int(rng.integers(0, len(df)))
    w0_path = df['path'].iloc[w0_idx]
    w0 = load_emb(w0_path).astype(np.float64)
    p, c0 = ss_gen(w0, scale, seed)
    c0_key = tuple(int(v) for v in np.asarray(c0).ravel())   # NO % Q -- raw 32-bit hashes
    soft = blocks_to_match < N_BLOCKS

    def matches_c0(cw):
        if not soft:
            return cw == c0_key
        return matching_blocks_exact(cw, c0_key) >= blocks_to_match

    counter = Counter()
    counted = no_majority = empty = decoded = match_c0 = 0
    for _, g in tqdm(df.groupby('id'), desc='recover', leave=False):
        votes = Counter()
        for pth in g['path']:
            if pth == w0_path:
                continue  # never score the enrolment against itself
            wp = load_emb(pth).astype(np.float64)
            r = tuple(int(v) for v in np.asarray(ss_rec(wp, p, scale=scale)).ravel())   # NO % Q
            votes[r] += 1
            decoded += 1
        total = sum(votes.values())
        if not total:
            empty += 1
            continue

        pool = group_codewords(votes, blocks_to_match, verbose=False, dim=DIM) if soft else votes

        if not elimination:
            for cw, cnt in pool.items():
                counter[cw] += cnt
                counted += cnt
                if matches_c0(cw):
                    match_c0 += cnt
        else:
            top_cw, top_count = pool.most_common(1)[0]
            majority = (top_count * 2 >= total) if soft else (top_count * 2 > total)
            if majority:
                counter[top_cw] += top_count
                counted += top_count
                if matches_c0(top_cw):
                    match_c0 += top_count
            else:
                no_majority += 1

    stats = {'counted': counted, 'match_c0': match_c0}
    return counter, stats


def estimate_scheme(df, scales, model, load_emb, seed, enrollments, elimination,
                    samples=0, blocks_to_match=N_BLOCKS, per_enrollment=False):
    """Sweep scales over `enrollments` draws for ourfe; writes/resumes a per-scheme
    CSV. Same layout/resume semantics as entropy_estimate_hard_threshold.estimate_scheme."""
    ss_gen, ss_rec = datagen_ourdef.ss_gen, datagen_ourdef.ss_rec
    csv_path = f'entropy_results/{model}_{SCHEME}_entropy{"_elim" if elimination else ""}{f"_samples{samples}" if samples > 0 else ""}_seed{seed}.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        prior = pd.read_csv(csv_path)
        records = prior.to_dict('records')
        if 'blocks_to_match' not in prior.columns:
            prior['blocks_to_match'] = N_BLOCKS
        if per_enrollment:
            enr = prior['enrollment'] if 'enrollment' in prior.columns else [seed] * len(prior)
            done_keys = {(float(s), int(b), int(en))
                         for s, b, en in zip(prior['scale'], prior['blocks_to_match'], enr)}
        else:
            done_keys = {(float(s), int(b))
                         for s, b in zip(prior['scale'], prior['blocks_to_match'])}
        print(f'[{SCHEME}] resuming with {len(done_keys)} keys already done')
    else:
        records, done_keys = [], set()

    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    if per_enrollment:
        pending = list(scales)
    else:
        pending = [s for s in scales if (float(s), int(blocks_to_match)) not in done_keys]
    scale_bar = tqdm(pending, desc=f'{model}/{SCHEME}{"_elim" if elimination else ""}')
    for scale in scale_bar:
        per = []
        for e in tqdm(range(enrollments), desc='enroll', leave=False):
            if per_enrollment and (float(scale), int(blocks_to_match), int(seed + e)) in done_keys:
                continue  # this draw is already in the CSV (resumed run)
            counter, stats = recover_histogram(ss_gen, ss_rec, df, scale,
                                               seed + e, load_emb, elimination, blocks_to_match)
            rec = {
                'blocks_to_match': blocks_to_match,
                'h_min': h_min(counter), 'h_shannon': h_shannon(counter),
                'h_collision': h_collision(counter), 'n_unique': len(counter),
                'counted': stats['counted'], 'match_c0': stats['match_c0'],
                'max_count': max(counter.values()) if counter else 0,
            }
            per.append(rec)
            if per_enrollment:
                records.append({'scheme': SCHEME, 'scale': scale,
                                'enrollment': int(seed + e),
                                'enrollments': enrollments, **rec})
                done_keys.add((float(scale), int(blocks_to_match), int(seed + e)))
                pd.DataFrame(records).to_csv(csv_path, index=False)
            if _cuda:
                torch.cuda.empty_cache()
        if not per_enrollment:
            agg = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
            records.append({'scheme': SCHEME, 'scale': scale,
                            'enrollments': enrollments, **agg})
            done_keys.add((float(scale), int(blocks_to_match)))
            pd.DataFrame(records).to_csv(csv_path, index=False)
        if per:
            scale_bar.set_postfix(scale=f'{scale:.1f}',
                                  h_min=f'{np.mean([d["h_min"] for d in per]):.2f}')
    return csv_path


def main():
    args = parse_args()
    df = build_population(args.dataset, args.model, args.splits, args.samples,
                          seed=args.seed, unprotected_dir=args.unprotected_dir)
    print(f'population={len(df)} embeddings, {df["id"].nunique()} identities, '
          f'scale={args.scale}, scheme={SCHEME}, blocks_to_match={args.blocks_to_match}, '
          f'elimination={"True" if args.elimination else "False"}, enrollments={args.enrollments}')

    load_emb = make_loader()  # disk read cached
    out = estimate_scheme(df, [args.scale], args.model, load_emb, args.seed,
                          args.enrollments, args.elimination, args.samples,
                          args.blocks_to_match, args.per_enrollment)
    print(f'[{SCHEME}] wrote {out}')


if __name__ == '__main__':
    main()
