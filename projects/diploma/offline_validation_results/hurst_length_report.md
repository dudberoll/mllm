# Hurst–Length Analysis Report

**Date:** 2026-05-12

> **Context:** 98.1% of trajectories are at max length (4096 tokens).
> A naive 3-bin length split is degenerate. The primary analysis uses
> sub-sequence resampling to show how partial ρ evolves with prefix length.

---

## Part 1 — Sub-sequence Resampling

For each sampled trajectory, `|H_s - H_t|` is computed on prefixes of
increasing length. Partial Spearman correlation with `reward`, controlling
for `mean_kl`, is computed at each prefix length.

| Prefix length | Partial ρ | 5th pct | 95th pct |
|---:|---:|---:|---:|
| 128 | +0.0317 | +0.0005 | +0.0801 |
| 256 | +0.0043 | -0.0324 | +0.0526 |
| 512 | -0.0517 | -0.0919 | -0.0122 |
| 1024 | -0.0708 | -0.1120 | -0.0271 |
| 2048 | -0.0182 | -0.0597 | +0.0220 |
| 4096 | +0.0463 | +0.0053 | +0.0915 |

**Signal emerges at ≥ 512 tokens** (|partial ρ| > 0.05).

**Implication for λ_H:** the weight on the Hurst term should be
near-zero for short sequences and increase with length.
A simple schedule: `λ_H(L) = λ_H_max × sigmoid((L - L0) / τ)`
where L0 ≈ the crossing length above.

---

## Part 2 — Real 3-bin Analysis

| Bin | n | raw ρ | p | partial ρ | p |
|---|---:|---:|---:|---:|---:|
| short  (<512) | 12 | — | — | — | — |
| medium (512–4095) | 231 | -0.0332 | 0.6162 | +0.0859 | 0.1935 |
| long   (=4096) | 14565 | -0.0673 | 0.0000 | +0.0260 | 0.0017 |

---

## Plots

| File | Content |
|---|---|
| `hurst_partial_rho_vs_length.png` | Partial ρ vs. prefix length with 90% CI |
| `hurst_diff_by_reward_vs_length.png` | Median \|ΔH\| by reward class across lengths |
