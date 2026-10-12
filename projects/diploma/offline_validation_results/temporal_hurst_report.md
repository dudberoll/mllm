# Temporal Hurst Analysis Report

**Date:** 2026-05-12

W=512 tokens per window, 8 windows per trajectory.

---

## Analysis 1 — Per-window partial ρ

| Window | tokens | partial ρ | p | sig |
|---:|---:|---:|---:|:---:|
| 0 | 0–512 | -0.0034 | 0.6841 |  |
| 1 | 512–1024 | -0.0083 | 0.3155 |  |
| 2 | 1024–1536 | -0.0086 | 0.2982 |  |
| 3 | 1536–2048 | -0.0105 | 0.2064 |  |
| 4 | 2048–2560 | +0.0083 | 0.3184 |  |
| 5 | 2560–3072 | +0.0006 | 0.9395 |  |
| 6 | 3072–3584 | -0.0030 | 0.7168 |  |
| 7 | 3584–4096 | -0.0024 | 0.7702 |  |

**Peak signal at window 3** (tokens 1536–2048), partial ρ = -0.0105

---

## Analysis 2 — Early (w0–6) vs Late (w7)

| Segment | partial ρ | p | n |
|---|---:|---:|---:|
| `early_std  (w0–6)` | +0.1032 * | 0.0000 | 14,565 |
| `late_diff  (w7)` | +0.1351 * | 0.0000 | 14,564 |
| `full_std   (w0–7)` | +0.1435 * | 0.0000 | 14,565 |

---

## Analysis 3 — Boxed / no-boxed stratification

| Group | n | raw ρ | p | partial ρ | p |
|---|---:|---:|---:|---:|---:|
| boxed | 3,538 | +0.2248 | 0.0000 | +0.1048 * | 0.0000 |
| no-boxed | 11,027 | +0.0674 | 0.0000 | +0.1305 * | 0.0000 |
| all | 14,565 | +0.1117 | 0.0000 | +0.1435 * | 0.0000 |

---

## Plots
| File | Content |
|---|---|
| `temporal_per_window.png` | Per-window raw and partial ρ bar chart |
| `temporal_early_vs_late.png` | Early vs late segment comparison |
| `temporal_boxed_split.png` | std_diff violin: boxed vs no-boxed subgroups |
