# Windowed Hurst Analysis Report

**Date:** 2026-05-12

Aggregations: `mean_abs` = mean|ΔH|, `max_abs` = max|ΔH|, `std_diff` = std(ΔH signed)

## Full Results Table

| W | Aggregation | n | raw ρ | p | partial ρ | p |
|---|---|---:|---:|---:|---:|---:|
| 256 | `mean_abs` | 14,796 | +0.0705 | 0.0000 | +0.0901 * | 0.0000 |
| 256 | `max_abs` | 14,796 | +0.0962 | 0.0000 | +0.1324 * | 0.0000 |
| 256 | `std_diff` | 14,796 | +0.1157 | 0.0000 | +0.1293 * | 0.0000 |
| 512 | `mean_abs` | 14,761 | +0.0415 | 0.0000 | +0.0784 * | 0.0000 |
| 512 | `max_abs` | 14,761 | +0.0689 | 0.0000 | +0.1288 * | 0.0000 |
| 512 | `std_diff` | 14,761 | +0.1064 | 0.0000 | +0.1380 * | 0.0000 |
| 1024 | `mean_abs` | 14,701 | -0.0022 | 0.7891 | +0.0591 * | 0.0000 |
| 1024 | `max_abs` | 14,701 | +0.0197 | 0.0169 | +0.1088 * | 0.0000 |
| 1024 | `std_diff` | 14,701 | +0.0774 | 0.0000 | +0.1275 * | 0.0000 |

**Best combination:** W=512, `std_diff`  → partial ρ = +0.1380 (p=0.0000)

## Plots

| File | Content |
|---|---|
| `windowed_hurst_heatmap.png` | Raw vs partial ρ for all W × aggregation |
| `windowed_hurst_best_boxplot.png` | Box plot for best combination |
| `windowed_hurst_trajectories_W256.png` | Per-window H_s / H_t examples at W=256 |
| `windowed_hurst_trajectories_W512.png` | Per-window H_s / H_t examples at W=512 |
| `windowed_hurst_trajectories_W1024.png` | Per-window H_s / H_t examples at W=1024 |
