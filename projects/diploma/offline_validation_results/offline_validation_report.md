# Offline Validation Report: Composite Regularizer Descriptors

**Date:** 2026-05-12  
**Records:** 14830  (valid Hurst: 14815)  
**Reward distribution:** -0.1 → 11734, 0.0 → 1, 0.9 → 3094, 1.0 → 1

---

## Check 1 — Predictive Validity

Spearman correlation of each descriptor with `individual_reward`,
significance via permutation test (1 000 shuffles).

| Descriptor | ρ | p (perm) | p (scipy) | Decision (\|ρ\|>0.15, p<0.05) |
|---|---:|---:|---:|:---|
| `var_diff` | -0.2454 | 0.0010 | 0.0000 | ✅ passes |
| `acf1_diff` | +0.0153 | 0.0609 | 0.0625 | ❌ fails |
| `hurst_diff` | -0.0620 | 0.0010 | 0.0000 | ❌ fails |

**Gate:** ≥2 descriptors must pass → ❌ GATE FAILED (1/3 pass)

---

## Check 2 — Descriptor Independence

Pairwise Spearman between descriptors.  
Values > 0.85 indicate collinearity (one could be dropped).

| | var\_diff | acf1\_diff | hurst\_diff |
|---|---:|---:|---:|
| `var_diff` | +1.0000 | +0.2737 | +0.4359 |
| `acf1_diff` | +0.2737 | +1.0000 | +0.3482 |
| `hurst_diff` | +0.4359 | +0.3482 | +1.0000 |

---

## Check 3 — Orthogonality to Baseline KL

Partial Spearman: does the descriptor carry signal beyond `mean_step_kl`?

| Descriptor | raw ρ | p | partial ρ | p | Decision |
|---|---:|---:|---:|---:|:---|
| `var_diff` | -0.2454 | 0.0000 | +0.3819 | 0.0000 | ✅ passes |
| `acf1_diff` | +0.0153 | 0.0625 | +0.1529 | 0.0000 | ✅ passes |
| `hurst_diff` | -0.0620 | 0.0000 | +0.0210 | 0.0104 | ❌ fails |

---

## Plots

| File | Content |
|---|---|
| `boxplots_by_reward.png` | Descriptor distribution split by reward class |
| `descriptor_corr_heatmap.png` | Pairwise Spearman heatmap |
| `permutation_test.png` | Null distributions vs. observed ρ |
| `raw_vs_partial_rho.png` | Raw vs. partial correlation bar chart |
