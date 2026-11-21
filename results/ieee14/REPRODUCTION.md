# Reproduction report - `ieee14`

Reproduced values (this repository) next to the values reported in the manuscript. Real-data quantities (TRTR, real-to-real baseline) are deterministic and must match; generator-dependent quantities vary with library version, hardware and RNG stream.

## Per-feature fidelity, CTGAN (Table 1)

| feature | HD (paper) | HD (repro) | KL (paper) | KL (repro) | KS (paper) | KS (repro) |
|---|---|---|---|---|---|---|
| V2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| V6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| V9 | 0.0921 | 0.0547 | 0.0497 | 0.0215 | 0.0651 | 0.0312 |
| Theta2 | 0.1341 | 0.1097 | 0.0819 | 0.0611 | 0.1174 | 0.0916 |
| Theta6 | 0.0926 | 0.0999 | 0.0339 | 0.0397 | 0.0821 | 0.0919 |
| Theta9 | 0.0879 | 0.1099 | 0.0287 | 0.0489 | 0.0560 | 0.1086 |
| P1 | 0.0981 | 0.0863 | 0.0449 | 0.0288 | 0.0904 | 0.0727 |
| P4 | 0.1079 | 0.0449 | 0.0491 | 0.0107 | 0.1124 | 0.0241 |
| Q4 | 0.1739 | 0.0646 | 0.1611 | 0.0171 | 0.1515 | 0.0416 |
| P5 | 0.1492 | 0.0621 | 0.1094 | 0.0167 | 0.1467 | 0.0379 |
| P7 | 0.0437 | 0.0581 | 0.0078 | 0.0129 | 0.0347 | 0.0611 |
| P9 | 0.0736 | 0.1741 | 0.0260 | 0.1256 | 0.0540 | 0.1467 |
| P10 | 0.0797 | 0.1023 | 0.0297 | 0.0591 | 0.0675 | 0.0817 |
| Q10 | 0.0268 | 0.1060 | 0.0032 | 0.0424 | 0.0319 | 0.0969 |
| P11 | 0.0540 | 0.1046 | 0.0115 | 0.0466 | 0.0588 | 0.0843 |
| P12 | 0.0413 | 0.0498 | 0.0064 | 0.0141 | 0.0375 | 0.0348 |
| Q12 | 0.0282 | 0.1311 | 0.0032 | 0.0704 | 0.0198 | 0.1350 |
| P13 | 0.0982 | 0.1299 | 0.0447 | 0.0987 | 0.0733 | 0.1171 |
| P14 | 0.0720 | 0.1278 | 0.0211 | 0.0654 | 0.0714 | 0.1165 |
| Q14 | 0.0362 | 0.1650 | 0.0071 | 0.1443 | 0.0244 | 0.1377 |
| LoadMargin | 0.0550 | 0.1198 | 0.0119 | 0.0545 | 0.0450 | 0.0985 |

## MDNN, CTGAN (Table 2)

| indicator | paper | repro |
|---|---|---|
| real_to_synth | 2.4455 | 2.4440 |
| synth_to_real | 2.4203 | 2.4406 |
| MDNN | 2.4329 | 2.4423 |
| real_to_real_loo | 2.2426 | 2.2426 |

## Model comparison (Table 3)

| model | mean_HD (paper) | mean_HD (repro) | max_HD (paper) | max_HD (repro) | mean_KL (paper) | mean_KL (repro) | mean_KS (paper) | mean_KS (repro) | max_KS (paper) | max_KS (repro) | MDNN (paper) | MDNN (repro) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctgan | 0.074 | 0.090 | 0.174 | 0.174 | 0.035 | 0.047 | 0.064 | 0.077 | 0.152 | 0.147 | 2.433 | 2.442 |
| tvae | 0.121 | 0.111 | 0.306 | 0.252 | 0.152 | 0.119 | 0.079 | 0.076 | 0.222 | 0.226 | 2.284 | 2.360 |
| copula | 0.013 | 0.021 | 0.020 | 0.051 | 0.001 | 0.003 | 0.009 | 0.013 | 0.015 | 0.042 | 2.259 | 2.244 |

## TSTR (Table 4)

| training_data | NRMSE_pct (paper) | NRMSE_pct (repro) | NMAE_pct (paper) | NMAE_pct (repro) |
|---|---|---|---|---|
| real (TRTR) | 0.70 | 0.74 | 0.50 | 0.49 |
| ctgan (TSTR) | 4.10 | 3.73 | 2.70 | 2.60 |
| tvae (TSTR) | 5.00 | 4.05 | 3.30 | 2.70 |
| copula (TSTR) | 2.00 | 1.91 | 1.30 | 1.32 |

## DCR (Table 5)

| dataset | mean (paper) | mean (repro) | p5 (paper) | p5 (repro) | min (paper) | min (repro) |
|---|---|---|---|---|---|---|
| real-to-real (baseline) | 2.243 | 2.243 | 1.628 | 1.628 | 1.078 | 1.078 |
| ctgan | 2.420 | 2.441 | 1.807 | 1.814 | 1.096 | 0.994 |
| tvae | 2.096 | 2.189 | 1.584 | 1.676 | 1.079 | 0.868 |
| copula | 2.267 | 2.244 | 1.655 | 1.641 | 0.931 | 0.917 |

## Missing-value completion (Figs. 3-4, quantified)

Distance between the completed feature and the original one (`std_ratio` = std of completed / std of original).

| feature | missing_pct | method | HD | KL | KS | std_ratio |
|---|---|---|---|---|---|---|
| V9 | 5 | local-mean | 0.0090 | 0.0003 | 0.0059 | 0.9858 |
| V9 | 5 | ctgan | 0.0047 | 0.0001 | 0.0018 | 0.9995 |
| V9 | 10 | local-mean | 0.0142 | 0.0008 | 0.0113 | 0.9762 |
| V9 | 10 | ctgan | 0.0061 | 0.0002 | 0.0044 | 1.0004 |
| V9 | 20 | local-mean | 0.0298 | 0.0036 | 0.0197 | 0.9468 |
| V9 | 20 | ctgan | 0.0123 | 0.0006 | 0.0080 | 0.9996 |
| V9 | 30 | local-mean | 0.0428 | 0.0075 | 0.0300 | 0.9240 |
| V9 | 30 | ctgan | 0.0161 | 0.0011 | 0.0137 | 1.0074 |
| P1 | 5 | local-mean | 0.0084 | 0.0003 | 0.0067 | 0.9858 |
| P1 | 5 | ctgan | 0.0049 | 0.0001 | 0.0045 | 1.0005 |
| P1 | 10 | local-mean | 0.0126 | 0.0006 | 0.0091 | 0.9780 |
| P1 | 10 | ctgan | 0.0086 | 0.0003 | 0.0076 | 1.0035 |
| P1 | 20 | local-mean | 0.0299 | 0.0037 | 0.0196 | 0.9456 |
| P1 | 20 | ctgan | 0.0215 | 0.0018 | 0.0170 | 1.0087 |
| P1 | 30 | local-mean | 0.0472 | 0.0092 | 0.0327 | 0.9152 |
| P1 | 30 | ctgan | 0.0291 | 0.0033 | 0.0251 | 1.0104 |
