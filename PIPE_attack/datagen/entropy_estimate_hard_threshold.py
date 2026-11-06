import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from leech_scales import DIM, N_BLOCKS, GENREP, Q, make_loader, count_matching_blocks

# Sparse-build memory budget for group_codewords (GB). The shared-block-count
# matrix needs ~est_pairs * 8 bytes; near-injective populations stay well under
# 1 GB, so this only trips on pathological mode-collapse (a block value shared by
# a huge fraction of codewords). Raise it (or cap --samples) if you hit it.
GROUP_MEM_BUDGET_GB = 40.0

# Above this many distinct codewords, group_codewords prefers a GPU that streams
# the pairwise shared-block comparison in tiles (O(n) memory, never materialising
# the ~n^2 adjacency) over the sparse CPU build -- exact same output, but it can
# handle block-collapsed populations whose sparse peak would exceed the budget.
GROUP_GPU_MIN_N = 20000
GROUP_GPU_TILE = 4096           # rows of the n x n adjacency streamed per tile

def parse_args():
    parser = argparse.ArgumentParser(description='Leech-FE recovered-secret entropy estimation')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--model', type=str, default='facenet')
    parser.add_argument('--unprotected-dir', type=str, default=None,
                        help='base dir of unprotected embeddings')
    parser.add_argument('--schemes', nargs='+', default=list(GENREP),
                        choices=list(GENREP), help='which schemes to estimate')
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
    parser.add_argument('--blocks-to-match', type=int, default=21)
    parser.add_argument('--scale', type=float, default=None,)
    return parser.parse_args()


def build_population(dataset, model, splits, samples, seed=0, unprotected_dir=None):
    """Return a DataFrame [path, id] of probe embeddings that exist on disk."""
    folder = unprotected_dir if unprotected_dir else f'../../datasets/pp/{dataset}/embeddings_{model}' # use the outer larger dataset
    img_id = pd.read_csv(f'../../datasets/pp/{dataset}/img_id_map_full.csv')
    if splits:
        img_id = img_id[img_id['label'].isin(splits)]

    rows = [(f"{folder}/{str(img[:-4]).zfill(6)}.npy", id_)
            for img, label, id_ in zip(img_id['img'], img_id['label'], img_id['id'])]
    df = pd.DataFrame(rows, columns=['path', 'id'])
    df = df[df['path'].map(os.path.exists)].reset_index(drop=True)
    if not len(df):
        raise FileNotFoundError(f'no embeddings on disk for {dataset}/{model} splits={splits}')

    if samples and samples < len(df):
        idx = np.sort(np.random.RandomState(seed).choice(len(df), samples, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    return df


# --- entropy of a codeword histogram (counts as a Counter) -----------------
def _probs(counter):
    c = np.array(list(counter.values()), dtype=np.float64)
    return c / c.sum()


def h_min(counter):
    """Min-entropy -log2(max p): the guessing-attack security metric."""
    if not counter:
        return 0.0
    p = _probs(counter)
    return float(-np.log2(p.max()))


def h_shannon(counter):
    if not counter:
        return 0.0
    p = _probs(counter)
    return float(-(p * np.log2(p)).sum())


def h_collision(counter):
    """Renyi-2 (collision) entropy -log2(sum p^2)."""
    if not counter:
        return 0.0
    p = _probs(counter)
    return float(-np.log2((p ** 2).sum()))

def _greedy_group_gpu(cws, counts, block_ids, blocks_to_match, tile=GROUP_GPU_TILE):
    """Exact GPU twin of the CPU greedy below, for populations whose n x n
    adjacency is too big to materialise on the host. It STREAMS the pairwise
    shared-block count in row tiles -- the only resident state is block_ids
    (n x n_blk, a few MB), counts and the weights/active vectors -- so memory is
    O(n) regardless of how block-collapsed the population is. Output is identical:
    same `shared >= blocks_to_match` adjacency (diagonal = n_blk), same
    highest-weight-seed-first ordering with lowest-index tie-breaking.

    Integer weights are accumulated through float32 matvecs, which is exact here
    because every partial sum is an integer <= total votes (<= population, well
    under 2**24).
    """
    import torch
    n, n_blk = block_ids.shape
    dev = torch.device('cuda')
    bids = torch.from_numpy(np.ascontiguousarray(block_ids, dtype=np.int32)).to(dev)
    counts_f = torch.from_numpy(counts.astype(np.float32)).to(dev)        # (n,)
    weights = torch.empty(n, dtype=torch.int64, device=dev)

    def tile_adj(rows_t):
        """(|rows|, n) float32 adjacency of the given rows vs all codewords."""
        A = bids[rows_t]                                                  # (r, n_blk)
        sc = torch.zeros((A.shape[0], n), dtype=torch.int16, device=dev)
        for k in range(n_blk):
            sc += (A[:, k:k + 1] == bids[:, k].unsqueeze(0)).to(torch.int16)
        return (sc >= blocks_to_match).to(torch.float32)

    # Initial weights[i] = votes in i's neighbourhood (incl. self), tiled over rows.
    idx_all = torch.arange(n, device=dev)
    for s in range(0, n, tile):
        rows_t = idx_all[s:s + tile]
        weights[s:s + rows_t.shape[0]] = (tile_adj(rows_t) @ counts_f).round().to(torch.int64)

    active = torch.ones(n, dtype=torch.bool, device=dev)
    neg = torch.tensor(-1, dtype=torch.int64, device=dev)
    grouped = Counter()
    while bool(active.any()):
        masked = torch.where(active, weights, neg)
        mx = masked.max()
        seed = int(torch.nonzero(masked == mx, as_tuple=False)[0].item())  # lowest index
        # seed's neighbourhood row: one (n, n_blk) compare, no tiling needed.
        sc_seed = (bids == bids[seed].unsqueeze(0)).sum(1)                  # (n,)
        members = torch.nonzero((sc_seed >= blocks_to_match) & active,
                                as_tuple=False).flatten()
        m_cpu = members.cpu().numpy()
        grouped[cws[seed]] = int(counts[m_cpu].sum())
        active[members] = False
        # weights -= sum over removed members m of counts[m] * adj(m, :), tiled.
        delta = torch.zeros(n, dtype=torch.float32, device=dev)
        for s in range(0, members.shape[0], tile):
            mt = members[s:s + tile]
            delta += counts_f[mt] @ tile_adj(mt)
        weights -= delta.round().to(torch.int64)
    return grouped


def group_codewords(counter, blocks_to_match, verbose=True, dim=DIM):
    """Greedily merge codewords sharing >= `blocks_to_match` identical blocks
    (each block is `dim` consecutive entries; defaults to the module DIM=24, but
    entropy_estimate_soft_threshold.py passes dim=1 for its 1-value-per-block keyed-hash sketch).
    Each round commits the group --
    a seed plus all not-yet-grouped codewords matching it -- that captures the
    most votes, then removes its members. Returns a Counter mapping each seed to
    its group's summed count, ready for the entropy helpers.

    The pairwise "shared-block count" matrix is built SPARSELY -- one-hot encode
    each block column (H_k: n x v_k, one 1 per row) and accumulate C = sum_k
    H_k @ H_k.T -- instead of the dense n x n grid. A near-injective population
    (n ~ 1e5 distinct recovered secrets, as in per-probe full-population runs)
    then stays in a few GB rather than the ~n^2 * 4 bytes (>100 GB) that OOM-kills
    the dense version. For large, block-collapsed populations whose sparse peak
    would still blow the budget, it dispatches to a GPU twin that streams the
    comparison in O(n) memory (`_greedy_group_gpu`). All three paths produce the
    identical result: the same C >= blocks_to_match adjacency (diagonal = n_blk,
    so singletons survive) and the same highest-weight-seed-first ordering with
    lowest-index tie-breaking.
    """
    cws = list(counter)
    n = len(cws)
    if n == 0:
        return Counter()
    counts = np.array([counter[cw] for cw in cws], dtype=np.int64)

    # Encode each codeword as N_BLOCKS compact block-ids. Codewords are integer
    # tuples, so block equality is exact and reduces to id equality -- replacing
    # the per-pair float-norm of count_matching_blocks with integer compares.
    blocks = np.asarray(cws, dtype=np.int64).reshape(n, -1, dim)   # (n, n_blk, dim)
    n_blk = blocks.shape[1]
    block_ids = np.empty((n, n_blk), dtype=np.int64)
    for k in range(n_blk):
        block_ids[:, k] = np.unique(blocks[:, k, :], axis=0, return_inverse=True)[1].ravel()
    del blocks

    # Cost guard: the sparse build's nnz is bounded by sum_k sum_buckets b^2
    # (pairs sharing a given block value). Near-injective data -> ~n per block; a
    # block value shared by a huge fraction of codewords (mode-collapse) is the
    # only thing that can blow this up, so surface it instead of silently thrashing.
    est_pairs = 0
    max_bucket = 0
    for k in range(n_blk):
        bc = np.bincount(block_ids[:, k])
        est_pairs += int((bc.astype(np.int64) ** 2).sum())
        max_bucket = max(max_bucket, int(bc.max()))
    approx_gb = est_pairs * 8 / 1e9     # ~int32 data + int32 index per stored pair

    # Prefer the GPU streamer for big populations -- it is O(n) memory, so it does
    # not care about approx_gb (the sparse peak that would otherwise OOM the host).
    use_gpu = False
    if n >= GROUP_GPU_MIN_N:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
    if verbose:
        print(f'[group_codewords] n={n} unique codewords, n_blk={n_blk}, '
              f'largest single-block bucket={max_bucket}, ~{est_pairs:,} candidate '
              f'pairs (~{approx_gb:.1f} GB sparse peak)'
              f'{" -> GPU streaming (O(n) host mem)" if use_gpu else ""}')
    if use_gpu:
        return _greedy_group_gpu(cws, counts, block_ids, blocks_to_match)
    if approx_gb > GROUP_MEM_BUDGET_GB:
        raise MemoryError(
            f'group_codewords would build ~{approx_gb:.1f} GB of sparse adjacency '
            f'(> GROUP_MEM_BUDGET_GB={GROUP_MEM_BUDGET_GB} GB); a single block value '
            f'is shared by {max_bucket} of {n} codewords. Use a CUDA GPU (auto for '
            f'n >= {GROUP_GPU_MIN_N}), cap the population with --samples, raise '
            f'--blocks-to-match, or raise GROUP_MEM_BUDGET_GB.')

    # Sparse shared-block-count matrix C = sum_k H_k @ H_k.T, where H_k is the
    # n x v_k one-hot of block k's value. (H_k @ H_k.T)[i,j] = 1 iff i,j share
    # block k; summing the n_blk columns yields the shared-block count. The
    # diagonal is n_blk (each node matches itself on every block), so singletons
    # survive the threshold below exactly as in the dense path.
    rows = np.arange(n)
    C = sparse.csr_matrix((n, n), dtype=np.int32)
    for k in range(n_blk):
        ids = block_ids[:, k]
        Hk = sparse.csr_matrix((np.ones(n, dtype=np.int32), (rows, ids)),
                               shape=(n, int(ids.max()) + 1))
        C = C + (Hk @ Hk.T)

    # Threshold on the stored counts (implicit zeros are < blocks_to_match anyway)
    # to get the boolean adjacency, then drop the now-zero entries -- avoids the
    # dense densification a sparse `C >= t` comparison would trigger.
    adj = C.tocsr().astype(np.int64)
    adj.data = (adj.data >= blocks_to_match).astype(np.int64)
    adj.eliminate_zeros()
    del C

    # Greedy, identical to the dense version but with sparse matvecs: pick the
    # highest-weight available group, commit it, then subtract the removed
    # members' votes from their neighbours' weights. adj is symmetric, so
    # adj[:,members] @ counts[members] == (adj[members,:].T) @ counts[members].
    active = np.ones(n, dtype=bool)
    weights = np.asarray(adj @ counts).ravel()          # votes in each node's nbhd
    grouped = Counter()
    while active.any():
        seed = int(np.argmax(np.where(active, weights, -1)))
        nbrs = adj.indices[adj.indptr[seed]:adj.indptr[seed + 1]]
        members = nbrs[active[nbrs]]
        grouped[cws[seed]] = int(counts[members].sum())
        active[members] = False
        weights -= np.asarray(adj[members, :].T @ counts[members]).ravel()
    return grouped



def recover_histogram(ss_gen, ss_rec, df, scale, seed, load_emb, elimination, blocks_to_match):
    """Build one enrolment's sketch, recover secrets across the population,
    and return (counter, stats). `counter` maps recovered secret (as a tuple)
    to its vote count.

    Four recovery modes, the cross-product of `elimination` and whether a soft
    block threshold is in play (`soft = blocks_to_match < N_BLOCKS`). In every
    mode each probe is decoded once with ss_rec; the modes differ only in how an
    identity's recovered secrets are reduced into histogram votes:

      baseline,    hard : every probe's recovered secret is counted as-is
                          (a plain population histogram).
      elimination, hard : per identity, the strict-majority secret (if one
                          holds > half the probes) casts a single vote; identities
                          with no majority are dropped.
      baseline,    soft : per identity, secrets are greedily merged into
                          soft-threshold clusters (>= blocks_to_match shared
                          blocks); every cluster contributes its summed votes.
      elimination, soft : per identity, the largest soft cluster casts a single
                          vote iff it holds >= half the identity's probes; else
                          the identity is dropped.

    The soft grouping is done WITHIN each identity (group_codewords on that
    identity's vote histogram), not globally on the final population histogram."""
    rng = np.random.default_rng(seed)
    w0_idx = int(rng.integers(0, len(df)))
    w0_path = df['path'].iloc[w0_idx]
    w0 = load_emb(w0_path).astype(np.float64)
    p, c0 = ss_gen(w0, scale, seed)
    c0_key = tuple((c0 % Q).tolist())
    soft = blocks_to_match < N_BLOCKS

    def matches_c0(cw):
        """Does a recovered secret count as recovering the enrolment secret c0?
        Exact tuple equality on the hard path, >= blocks_to_match shared blocks
        on the soft path (the same criterion the grouping merges on)."""
        if not soft:
            return cw == c0_key
        return count_matching_blocks(np.array(cw), np.array(c0_key)) >= blocks_to_match

    counter = Counter()
    counted = no_majority = empty = decoded = match_c0 = 0
    # One identity at a time: an identity's probes vote for recovered secrets,
    # then the mode decides what (if anything) those votes contribute.
    for _, g in tqdm(df.groupby('id'), desc='recover', leave=False):
        votes = Counter()
        for pth in g['path']:
            if pth == w0_path:
                continue  # never score the enrolment against itself
            wp = load_emb(pth).astype(np.float64)
            r = tuple((ss_rec(wp, p, scale=scale) % Q).tolist())
            votes[r] += 1
            decoded += 1
        total = sum(votes.values())
        if not total:
            empty += 1
            continue

        # Soft modes collapse near-identical codewords into clusters first; the
        # per-identity histogram is tiny, so silence group_codewords' diagnostics.
        pool = group_codewords(votes, blocks_to_match, verbose=False) if soft else votes

        if not elimination:
            # Baseline: keep every (merged) cluster, counted with its votes.
            for cw, cnt in pool.items():
                counter[cw] += cnt
                counted += cnt
                if matches_c0(cw):
                    match_c0 += cnt
        else:
            # Elimination: the dominant codeword/cluster survives iff it is a
            # majority of the identity's probes, and casts its full vote count
            # into the histogram (the discarded codewords contribute nothing).
            # Soft uses >= half (a cluster can tie); hard keeps the strict > half
            # on exact codewords.
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


def estimate_scheme(scheme_name, df, scales, model, load_emb, seed, enrollments, elimination, samples=0, blocks_to_match=21, per_enrollment=False):
    """Sweep scales for one scheme over `enrollments` draws. By default writes a
    single mean-over-enrolments row per scale; with `per_enrollment` it writes
    one row per draw (an `enrollment` column carries the draw's seed). Writes/
    resumes a per-scheme CSV."""
    ss_gen, ss_rec = GENREP[scheme_name]
    csv_path = f'entropy_results/{model}_{scheme_name}_entropy{"_elim" if elimination else ""}{f"_samples{samples}" if samples > 0 else ""}_seed{seed}.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        prior = pd.read_csv(csv_path)
        records = prior.to_dict('records')
        # Unique key is (scale, blocks_to_match) for the aggregated layout, and
        # (scale, blocks_to_match, enrollment) when each draw is its own row so a
        # long multi-enrolment run resumes mid-sweep. Older CSVs without the
        # column are treated as blocks_to_match=21 (the no-grouping baseline).
        if 'blocks_to_match' not in prior.columns:
            prior['blocks_to_match'] = 21
        if per_enrollment:
            enr = prior['enrollment'] if 'enrollment' in prior.columns else [seed] * len(prior)
            done_keys = {(float(s), int(b), int(en))
                         for s, b, en in zip(prior['scale'], prior['blocks_to_match'], enr)}
        else:
            done_keys = {(float(s), int(b))
                         for s, b in zip(prior['scale'], prior['blocks_to_match'])}
        print(f'[{scheme_name}] resuming with {len(done_keys)} keys already done')
    else:
        records, done_keys = [], set()

    # The GPU grouper (blocks_to_match != 21 on a large, block-collapsed draw)
    # leaves its tensors in torch's caching allocator. Across enrolments that is
    # reused, not leaked, but the reserved memory is never handed back to the OS;
    # a differently-sized next draw can then fragment-OOM on a card the first
    # draw already nearly filled. Empty the cache between draws so each enrolment
    # starts clean. Guarded so CPU-only / torch-less environments pay nothing.
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    # In aggregated mode a fully-done scale is skipped wholesale; in
    # per-enrollment mode resume is per draw, so keep every scale and skip
    # individual draws below.
    if per_enrollment:
        pending = list(scales)
    else:
        pending = [s for s in scales if (float(s), int(blocks_to_match)) not in done_keys]
    scale_bar = tqdm(pending, desc=f'{model}/{scheme_name}{"_elim" if elimination else ""}')
    for scale in scale_bar:
        # One row of entropy metrics per enrolment draw; averaged into a single
        # row unless per_enrollment keeps each draw.
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
                # Flush after every draw so a long multi-enrolment run is
                # checkpointed incrementally and resumes at draw granularity.
                records.append({'scheme': scheme_name, 'scale': scale,
                                'enrollment': int(seed + e),
                                'enrollments': enrollments, **rec})
                done_keys.add((float(scale), int(blocks_to_match), int(seed + e)))
                pd.DataFrame(records).to_csv(csv_path, index=False)
            # recover_histogram has returned, so the grouper's GPU tensors are
            # already dereferenced; release them to the OS before the next draw.
            if _cuda:
                torch.cuda.empty_cache()
        if not per_enrollment:
            agg = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
            records.append({'scheme': scheme_name, 'scale': scale,
                            'enrollments': enrollments, **agg})
            done_keys.add((float(scale), int(blocks_to_match)))
            pd.DataFrame(records).to_csv(csv_path, index=False)
        if per:
            scale_bar.set_postfix(scale=f'{scale:.1f}',
                                  h_min=f'{np.mean([d["h_min"] for d in per]):.2f}')
    return csv_path


def main():
    args = parse_args()
    df = build_population(args.dataset, args.model, args.splits, args.samples, seed=args.seed, unprotected_dir=args.unprotected_dir)
    print(f'population={len(df)} embeddings, {df["id"].nunique()} identities, '
          f'scale={args.scale}, schemes={args.schemes}, '
          f'elimination={"True" if args.elimination else "False"}, enrollments={args.enrollments}')

    load_emb = make_loader()  # disk read cached, shared across schemes
    for scheme_name in args.schemes:
        out = estimate_scheme(scheme_name, df, [args.scale], args.model, load_emb,
                              args.seed, args.enrollments, args.elimination, args.samples,
                              args.blocks_to_match, args.per_enrollment)
        print(f'[{scheme_name}] wrote {out}')


if __name__ == '__main__':
    main()
