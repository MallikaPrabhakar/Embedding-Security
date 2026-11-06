#!/usr/bin/env python3
"""Checkpoint (training step) vs ASR for ALL PIPE-attack schemes.

Reads the per-checkpoint eval outputs directly under PIPE_attack/results and
produces two plots plus printed tables for both ASR notions:

  L2 ASR   recon matches the original face: fraction of the 1000 reconstructions
           with Facenet512 euclidean_l2 distance < THRESHOLD.
           source: <ckpt>/<scheme>/.../test_distances.npy   (scripts/evaluation.py)

  rep ASR  recovers the protected codeword (FE secret):
           leech/e8  -> fraction matching >= threshold lattice blocks
                        source: test_protected_blocks_<tag>.npy (scripts/eval_protected.py)

Per request the plot caps facialfe_e8 and unprotected at 25k steps and every
other scheme at 50k; the printed tables show every checkpoint on disk.

    python PIPE_attack/ckpt_vs_asr.py
"""
import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

ROOT = os.getenv("PROJECT_ROOT")
print(f"PROJECT_ROOT: {ROOT}")
PIPE_DIR = os.path.normpath(os.path.join(ROOT, "PIPE_attack"))
RESULTS = os.path.normpath(os.path.join(PIPE_DIR, "results"))
OUT_L2 = os.path.join(PIPE_DIR, "ckpt_vs_asr_l2.png")     # L2 ASR plot
OUT_REP = os.path.join(PIPE_DIR, "ckpt_vs_asr_rep.png")   # rep ASR plot

L2_THRESHOLD = 1.1027              # Facenet512 euclidean_l2 (scripts/evaluation.py)
RAND_MEAN, RAND_STD = 1.01, 0.52  # random-guessing L2 ASR [%] (1000 samples)
# rep-ASR random-guess floor (mean%, std%) per scheme (celeba test, random_guessing_results.csv)
# unprotected has no FE secret => its rep floor == the L2 recon floor (RAND_MEAN, RAND_STD)
RAND_REP = {"Unprotected": (RAND_MEAN, RAND_STD),
            "Facial-FE e8": (27.5, 8.8), "Facial-FE leech": (3.0, 1.1),
            r"Facial-FE leech randomized": (3.0, 1.1), "Snapper-FE (ours)": (4.1, 1.5)}

# display name, facenet-dir glob (rel to RESULTS), rep blocks filename (None=>no rep),
# rep threshold, n_blocks/coords, plot step cap, color, marker
SCHEMES = [
    ("Unprotected",        "ckpt_*/unprotected/celeba/facenet",       None,                                                None, None, 25000, "#8c564b", "x"),
    ("Facial-FE e8",        "ckpt_*/facialfe_e8/celeba/facenet",               "test_protected_blocks_e8.npy",                        64,  64, 25000, "#d62728", "D"),
    ("Facial-FE leech",     "ckpt_*/facialfe_leech/celeba/facenet",            "test_protected_blocks_facialfe_leech.npy",            21,  21, 50000, "#1f77b4", "o"),
    (r"Facial-FE leech randomized", "ckpt_*/facialfe_leech_randomised/celeba/facenet", "test_protected_blocks_facialfe_leech_randomised.npy", 21,  21, 50000, "#ff7f0e", "s"),
    ("Snapper-FE (ours)",      "ckpt_*/ourfe/celeba/facenet",                      "test_protected_blocks_ourfe.npy",                      4,  21, 50000, "#17becf", "*"),
]


def _step(path):
    m = re.search(r"ckpt_(\d+)", path)
    return int(m.group(1)) if m else None


def collect():
    """-> {scheme: {step: {'l2':pct or None, 'rep':pct or None}}}"""
    data = {}
    for name, base, blk_file, thr, _nb, _cap, *_ in SCHEMES:
        per = {}
        for d in sorted(glob.glob(os.path.join(RESULTS, base))):
            step = _step(d)
            if step is None:
                continue
            l2 = rep = None
            dp = os.path.join(d, "test_distances.npy")
            if os.path.exists(dp):
                l2 = float(np.mean(np.load(dp) < L2_THRESHOLD)) * 100.0
            if blk_file is not None:
                bp = os.path.join(d, blk_file)
                if os.path.exists(bp):
                    rep = float(np.mean(np.load(bp) >= thr)) * 100.0
            elif blk_file is None:  # unprotected: no FE secret => full recon == full exposure (ceiling)
                rep = l2
            per[step] = {"l2": l2, "rep": rep}
        data[name] = per
    return data


def print_table(data, key, title):
    steps = sorted({s for per in data.values() for s in per})
    if key == "l2":
        names = [s[0] for s in SCHEMES]
    else:
        names = [s[0] for s in SCHEMES
                 if any(v.get("rep") is not None for v in data[s[0]].values())]
    head = "ckpt".rjust(7) + "".join(n.rjust(max(len(n), 6) + 2) for n in names)
    print(f"\n{title}")
    print(head)
    print("-" * len(head))
    for st in steps:
        cells = []
        for n in names:
            v = data[n].get(st, {}).get(key)
            cells.append("-" if v is None else f"{v:.1f}")
        print(str(st).rjust(7) + "".join(c.rjust(max(len(n), 6) + 2)
                                          for c, n in zip(cells, names)))


def plot(data):
    """Two SEPARATE figures: L2 ASR -> OUT_L2, rep ASR -> OUT_REP."""
    all_steps = set()
    SUB = "celeba, facenet, test split, 1000 samples"

    # ---------- figure 1: L2 ASR (recon matches original) ----------
    figL, axL = plt.subplots(figsize=(8, 4.5))
    for name, _b, blk_file, thr, nb, cap, color, marker in SCHEMES:
        pts = sorted((s, v["l2"]) for s, v in data[name].items()
                     if v["l2"] is not None and s <= cap)
        if pts:
            xs, ys = zip(*pts); all_steps.update(xs)
            axL.plot([x / 1000 for x in xs], ys, marker=marker, color=color, lw=2, ms=6, alpha=0.8, label=name)
    # random-guessing reference: mean line + ±1std band
    axL.axhline(RAND_MEAN, color="black", ls=":", lw=2.5, alpha=0.8, zorder=5,
                label=f"random guess ({RAND_MEAN}±{RAND_STD}%)")
    axL.axhspan(max(0, RAND_MEAN - RAND_STD), RAND_MEAN + RAND_STD, color="gray", alpha=0.3, zorder=4)
    # axL.set_title(f"PIPE attack — L2 ASR (recon matches original)\n"
    #               f"Facenet512 euclidean_l2 < {L2_THRESHOLD}  [{SUB}]", fontsize=15)
    axL.set_ylabel(r"Attack Success Rate ($\ell_2$) [%]", fontsize=15)
    axL.set_xlabel(r"Training checkpoints ($10^3$)", fontsize=15)
    axL.tick_params(labelsize=15)
    axL.grid(True, alpha=0.3); axL.legend(fontsize=10, loc="upper right", framealpha=0.4)

    # ---------- figure 2: rep ASR (recovers protected codeword) ----------
    figR, axR = plt.subplots(figsize=(8, 4.5))
    for name, _b, blk_file, thr, nb, cap, color, marker in SCHEMES:
        pts = sorted((s, v["rep"]) for s, v in data[name].items()
                     if v["rep"] is not None and s <= cap)
        if not pts:
            continue
        xs, ys = zip(*pts); all_steps.update(xs)
        if blk_file is None:
            tag = "no FE secret; =recon"
        else:
            tag = f"={nb}/{nb}" if thr == nb else f"≥{thr}/{nb}"
        axR.plot([x / 1000 for x in xs], ys, marker=marker, color=color, lw=2, ms=6, alpha=0.8, label=f"{name} ({tag})")
        # per-scheme random-guess floor: mean dotted line + ±1std band (scheme color)
        if name in RAND_REP:
            mu, sd = RAND_REP[name]
            axR.axhline(mu, color=color, ls=":", lw=2.5, alpha=0.8, zorder=5)
            axR.axhspan(max(0, mu - sd), mu + sd, color=color, alpha=0.15, zorder=4)
    # axR.set_title(f"PIPE attack — rep ASR (recovers protected codeword)\n"
    #               f"leech/e8: blocks  [{SUB}]", fontsize=15)
    axR.set_ylabel("Attack Success Rate (Rep) [%]", fontsize=15)
    axR.set_xlabel(r"Training checkpoints ($10^3$)", fontsize=15)
    axR.tick_params(labelsize=15)
    axR.plot([], [], color="black", ls=":", lw=2.5, label="random guess (per scheme)")
    axR.grid(True, alpha=0.3); axR.legend(fontsize=10, loc="upper right", framealpha=0.4)

    if all_steps:
        mx = max(all_steps) / 1000  # ticks in units of 10^3 steps
        for ax in (axL, axR):
            ax.set_xticks(np.arange(5, mx + 0.1, 5)); ax.tick_params(axis="x", rotation=45, labelsize=13)

    figL.tight_layout(); figL.savefig(OUT_L2, dpi=150)
    figR.tight_layout(); figR.savefig(OUT_REP, dpi=150)
    print(f"\nsaved {OUT_L2}\nsaved {OUT_REP}")


if __name__ == "__main__":
    data = collect()
    print_table(data, "l2", f"L2 ASR [%]  (Facenet512 euclidean_l2 < {L2_THRESHOLD})")
    print_table(data, "rep", "rep ASR [%]  (codeword recovery; per-scheme threshold)")
    plot(data)
