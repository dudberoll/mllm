# Confounder Analysis Report

**Date:** 2026-05-12  
**Feature:** windowed `std_diff`  (W=512, best from windowed_hurst_analysis.py)

---

## Check 1 — Sign Sanity

Spearman ρ(std_diff, reward) = **+0.1064**  
→ higher std_diff → higher reward  

> reward coding: −0.1 = incorrect, 0.9 = correct  (bigger is better)

---

## Check 3 — Per-run Partial ρ

| Run | n | raw ρ | p | partial ρ | p |
|---|---:|---:|---:|---:|---:|
| `onpolicy-deepmath-100steps-20260304-010826` | 1,998 | +0.0825 | 0.0002 | +0.3328 * | 0.0000 |
| `onpolicy-deepmath-g64-gs2-s50-20260304-114700` | 6,382 | +0.0787 | 0.0000 | +0.1411 * | 0.0000 |
| `onpolicy-deepmath-g64-gs2-s50-nores-20260304-` | 6,381 | +0.0976 | 0.0000 | +0.1578 * | 0.0000 |

---

## Check 4 — Full Confounders

| Control set | partial ρ | p |
|---|---:|---:|
| raw (no control) | +0.1064 * | 0.0000 |
| | KL | +0.1380 * | 0.0000 |
| | KL, n_active | +0.1415 * | 0.0000 |
| | KL, n_active, var_diff | +0.1499 * | 0.0000 |
| | KL, n_active, var_diff, acf1 | +0.1505 * | 0.0000 |

---

## Plots
| File | Content |
|---|---|
| `confounder_sign_check.png` | Violin + KDE of std_diff by reward class |
| `confounder_scatter_shape.png` | ECDF + Q–Q plot |
| `confounder_per_run.png` | Per-run partial ρ bar chart |
| `confounder_full.png` | Partial ρ as confounders are added progressively |
