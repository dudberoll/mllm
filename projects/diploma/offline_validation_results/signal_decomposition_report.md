# Signal Decomposition Report

**Date:** 2026-05-12

W=512, 8 windows, MIN_WIN=6 valid pairs required.
Partial control: KL, n_active, var_diff, acf1_diff (all simultaneously).

---

## Candidate Comparison

| Candidate | n | raw ρ | p | partial ρ | p |
|---|---:|---:|---:|---:|---:|
| `std_diff` | 14,564 | +0.1118 | 0.0000 | +0.1518 * | 0.0000 |
| `mean_abs` | 14,564 | +0.0426 | 0.0000 | +0.1345 * | 0.0000 |
| `corr_st` | 14,564 | +0.1160 | 0.0000 | +0.0750 * | 0.0000 |
| `std_s` | 14,564 | +0.2011 | 0.0000 | +0.2739 * | 0.0000 |
| `std_t` | 14,564 | +0.2085 | 0.0000 | +0.2884 * | 0.0000 |

---

## Pairwise Candidate Correlations
(n=14,564 records with all candidates finite)

| | `std_diff` | `mean_abs` | `corr_st` | `std_s` | `std_t` |
|---|---:|---:|---:|---:|---:|
| `std_diff` | +1.000 | +0.732 | -0.571 | +0.152 | +0.130 |
| `mean_abs` | +0.732 | +1.000 | -0.389 | +0.153 | +0.124 |
| `corr_st` | -0.571 | -0.389 | +1.000 | +0.604 | +0.600 |
| `std_s` | +0.152 | +0.153 | +0.604 | +1.000 | +0.816 |
| `std_t` | +0.130 | +0.124 | +0.600 | +0.816 | +1.000 |

---

## Plots
| File | Content |
|---|---|
| `decomp_partial_rho.png` | Raw vs partial ρ for all candidates |
| `decomp_pairwise_heatmap.png` | Pairwise Spearman between candidates |
