# GAN-attack results on CelebA / FaceNet (per-image embedding ASR)

ASR-L2: a reconstruction succeeds if the FaceNet *evaluation* embedding's euclidean_l2
(unit-normalised) to the target identity's train-time embedding is < 1.1027 (the Facenet512
threshold).

Classifier acc is the scheme's trained-classifier test accuracy.

| scheme | classifier acc (%) | GMI ASR-L2 | DMI ASR-L2 |
|---|---|---|---|
| unprotected | 79.52 | 85.02 | 94.54 |
| facialfe_e8 | 87.40 | 95.28 | 97.88 |
| facialfe_leech | 53.29 | 72.96 | 91.62 |
| facialfe_leech_randomised | 0.23 | 4.72 | 5.54 |
| ourfe | 0.17 | 3.08 | 3.64 |

To (re)generate the table from the saved reconstructions:

```bash
bash scripts/aggregate_results.sh 0          # writes RESULTS_SUMMARY.md
```
