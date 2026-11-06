# Recovered-secret entropy results

Min-entropy (`h_min`, bits) of the recovered-secret histogram for each FE scheme,
under the **baseline** (per-probe) and **elimination** (per-identity majority) adversaries,
**hard** matching unless noted. Higher `h_min` = more secure (harder to guess the recovered
secret). CSVs live alongside this file (`facenet_<scheme>_entropy[_elim]*_seed<seed>.csv`).

## seed-111 — full population (202,602 embeddings, 10,177 identities), 10 enrolments

Reported as mean ± std over the 10 enrolment draws (seeds 111–120).

| Scheme | scale | h_min baseline | h_min elim |
|--------|------:|---------------:|-----------:|
| **ourfe** (our defense; soft, blocks=4) | 144.64 | **12.50 ± 0.00** | **12.45 ± 0.00** |
| facialfe (hard) | 1461.76 | 4.98 ± 0.50 | 0.06 ± 0.05 |
| facialfe_randomised (hard) | 1461.76 | 4.98 ± 0.50 | 0.06 ± 0.05 |
| facialfe_e8 (hard) | 7.25 | 3.62 ± 0.77 | 0.024 ± 0.014 |

## Takeaways

- **ourfe (our defense) is secure**: `h_min ≈ 12.5 bits`, the clever (elimination) adversary barely dents it. ourfe's block-wise hash implemented by the per-block keyed BLAKE2b MAC is
  injective per 24-d block, so the recovered-secret entropy equals that of the decoded Leech codeword.
  The std is exactly 0 because decode recovery is invariant to the enrolment and the per-block keys
  (a different seed only relabels each block bijectively).
- **facialfe / facialfe_randomised leak** (`h_min ≈ 5.0` baseline) and **collapse to ≈ 0.06 bits**
  under the elimination adversary — effectively guessable. The two are identical, as their
  construction predicts (the random codeword cancels).
- **facialfe_e8 leaks the most**: lowest baseline `h_min ≈ 3.62` (a large ~18.8k-probe cluster
  recovers the mode codeword) and the most catastrophic elimination collapse (`≈ 0.024 bits`).

Security ordering (most → least secure): **ourfe ≫ facialfe ≈ facialfe_randomised > facialfe_e8**.


## Reproducing these results

Run `./entropy.sh` from `PIPE_attack/datagen/` (it `cd`s there itself). The single driver
computes every scheme's entropy in parallel (baseline + elimination each), streaming one
log per job to `entropy_results/logs/` and writing per-enrolment CSVs into
`entropy_results/facenet_<scheme>_entropy[_elim][_samples<N>]_seed<seed>.csv` (one row per
enrolment draw; aggregate as mean ± std over the rows). All use facenet embeddings over the
celeba population (`train`/`test_seen`/`test`/`test2` splits).

| Row(s) in the tables above | entropy driver |
|---|---|
| facialfe, facialfe_randomised, facialfe_e8 | `entropy_estimate_hard_threshold.py` — HARD thresholding (exact `--blocks-to-match 21`) over the lattice secure-sketch codeword |
| **ourfe** | `entropy_estimate_soft_threshold.py` — SOFT thresholding (`--blocks-to-match 4`) over ourfe's native keyed-hash sketch (no mod-Q, 21 blocks of dim 1, exact integer block match) |

`entropy_estimate_hard_threshold.py` builds the recovered-secret histogram per scheme and reports
`h_min`.
`entropy_estimate_soft_threshold.py` is its twin specialised to ourfe's keyed-hash sketch (see that
file's module docstring for the two differences). The leech-FE schemes decode each
embedding to the Leech lattice (`leech_helper.py`); ourfe's gen/rep lives in
`datagen_ourdef.py`.
