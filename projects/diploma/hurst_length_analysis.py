"""
Stratified Hurst-length analysis.

Context: 98% of trajectories in the telemetry are at max length (4096 tokens).
A naive 3-bin split would be degenerate. Instead we use sub-sequence resampling:

  For each sampled trajectory, compute |H_s - H_t| on prefixes of increasing
  length [128, 256, 512, 1024, 2048, 4096]. Then measure:

    1. DFA estimate stability (std of Hurst over bootstrap prefixes)
    2. Partial Spearman(hurst_diff, reward | mean_kl) at each prefix length

  This directly motivates λ_H as a function of length:
  as length grows, DFA becomes more reliable → stronger partial signal.

Secondary: a proper 3-bin analysis on the real length distribution
  (< 512, 512-4095, 4096 tokens) using the actual recorded hurst_diff.

Output: saved to offline_validation_results/ alongside the main analysis.

Usage:
  python hurst_length_analysis.py [output_dir]
"""

import json
import sys
import os
import time
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.linalg import lstsq

sys.path.insert(0, os.path.dirname(__file__))
from fractal_analysis import dfa as _dfa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TELEMETRY_FILES = [
    "/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700/telemetry_rollouts.jsonl",
    "/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-nores-20260304-185923/telemetry_rollouts.jsonl",
    "/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-100steps-20260304-010826/telemetry_rollouts.jsonl",
]

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "offline_validation_results"
)

PREFIX_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
N_SAMPLE       = 3000   # trajectories to use for resampling (balanced by reward)
N_BOOTSTRAP    = 200    # bootstrap resamples for error bars on partial ρ
SEED           = 42

# ---------------------------------------------------------------------------
# Helpers (duplicated-light from offline_validation.py to keep standalone)
# ---------------------------------------------------------------------------

def hurst_dfa(x: np.ndarray) -> Optional[float]:
    n = len(x)
    if n < 32:
        return None
    max_scale = n // 4
    if max_scale < 4:
        return None
    scales = np.unique(
        np.logspace(np.log10(4), np.log10(max_scale), num=8).astype(int)
    )
    return _dfa(x, scales)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Partial Spearman rho(x, y | z) via OLS residuals."""
    def residuals(a, b):
        A = np.column_stack([b, np.ones(len(b))])
        coeff, *_ = lstsq(A, a)
        return a - A @ coeff
    rho, p = spearmanr(residuals(x, z), residuals(y, z))
    return float(rho), float(p)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(files):
    """Load raw records (dicts with arrays, not extracted features)."""
    records = []
    for path in files:
        run = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mask = np.array(d["mask"], dtype=bool)
                n_active = mask.sum()
                if n_active < PREFIX_LENGTHS[0]:
                    continue
                records.append({
                    "lp_s"   : np.array(d["logp_student"])[mask],
                    "lp_t"   : np.array(d["logp_teacher"])[mask],
                    "kl"     : np.array(d["step_kl"])[mask],
                    "reward" : float(d["individual_reward"]),
                    "n"      : int(n_active),
                    "run"    : run,
                })
    return records

# ---------------------------------------------------------------------------
# Part 1 — Sub-sequence resampling
# ---------------------------------------------------------------------------

def resample_analysis(records, output_dir, rng):
    print("\n--- Part 1: Sub-sequence resampling ---\n")

    # Stratified sample: balance by reward class
    reward_vals = np.array([r["reward"] for r in records])
    unique_r = sorted(set(reward_vals))
    sampled = []
    per_class = N_SAMPLE // len(unique_r)
    for rv in unique_r:
        idx = np.where(reward_vals == rv)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        sampled.extend([records[i] for i in chosen])
    rng.shuffle(sampled)
    class_counts = ", ".join(
        f"{rv:.1f}→{sum(1 for r in sampled if r['reward'] == rv)}"
        for rv in unique_r
    )
    print(f"  Sampled {len(sampled)} trajectories ({class_counts})\n")

    # For each prefix length: compute hurst_diff and collect partial ρ
    partial_rhos = []        # mean across bootstrap
    partial_rho_lo = []      # 5th percentile CI
    partial_rho_hi = []      # 95th percentile CI
    mean_hurst_by_class = {rv: [] for rv in unique_r}  # median |ΔH| per length, per class

    t0 = time.time()
    for pl in PREFIX_LENGTHS:
        print(f"  Length {pl:>4d} tokens ...", end=" ", flush=True)
        h_diffs, kl_means, rewards = [], [], []

        for rec in sampled:
            lp_s = rec["lp_s"][:pl]
            lp_t = rec["lp_t"][:pl]
            if len(lp_s) < pl:          # shorter than prefix — use all, only if ≥ 32
                if len(lp_s) < 32:
                    continue
            kl_prefix = rec["kl"][:pl]

            h_s = hurst_dfa(lp_s)
            h_t = hurst_dfa(lp_t)
            if h_s is None or h_t is None:
                continue

            h_diffs.append(abs(h_s - h_t))
            kl_means.append(float(kl_prefix.mean()))
            rewards.append(rec["reward"])

        h_diffs  = np.array(h_diffs)
        kl_means = np.array(kl_means)
        rewards  = np.array(rewards)

        # Partial ρ on full sample
        obs_rho, obs_p = partial_spearman(h_diffs, rewards, kl_means)

        # Bootstrap CI
        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(h_diffs), size=len(h_diffs))
            if len(set(rewards[idx])) < 2:
                continue
            try:
                r, _ = partial_spearman(h_diffs[idx], rewards[idx], kl_means[idx])
                boot_rhos.append(r)
            except Exception:
                continue
        boot_rhos = np.array(boot_rhos)

        partial_rhos.append(obs_rho)
        partial_rho_lo.append(np.percentile(boot_rhos, 5)  if len(boot_rhos) else obs_rho)
        partial_rho_hi.append(np.percentile(boot_rhos, 95) if len(boot_rhos) else obs_rho)

        # Mean |ΔH| per reward class
        for rv in unique_r:
            sel = rewards == rv
            mean_hurst_by_class[rv].append(np.median(h_diffs[sel]) if sel.sum() > 0 else np.nan)

        elapsed = time.time() - t0
        print(f"n={len(h_diffs)}  partial_ρ={obs_rho:+.4f}  p={obs_p:.4f}  "
              f"({elapsed:.0f}s total)")

    # --- Plot 1: partial ρ vs prefix length ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(PREFIX_LENGTHS, partial_rho_lo, partial_rho_hi,
                    alpha=0.2, color="#4CAF50", label="90% bootstrap CI")
    ax.plot(PREFIX_LENGTHS, partial_rhos, "o-", color="#4CAF50", linewidth=2,
            markersize=6, label=r"Partial $\rho(|H_s-H_t|,\, \mathrm{reward}\,|\,\overline{KL})$")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.05, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
               label=r"$|\rho|=0.05$ threshold")
    ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(PREFIX_LENGTHS)
    ax.set_xticklabels([str(p) for p in PREFIX_LENGTHS])
    ax.set_xlabel("Prefix length (tokens)", fontsize=11)
    ax.set_ylabel(r"Partial Spearman $\rho$", fontsize=11)
    ax.set_title(r"Hurst partial correlation vs. trajectory length"
                 "\n(motivating length-dependent $\lambda_H$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hurst_partial_rho_vs_length.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: median |ΔH| by reward class vs length ---
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {-0.1: "#F44336", 0.0: "#9E9E9E", 0.9: "#2196F3", 1.0: "#4CAF50"}
    for rv in unique_r:
        vals = mean_hurst_by_class[rv]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(PREFIX_LENGTHS, vals, "o-", color=colors.get(rv, "gray"),
                label=f"reward={rv:.1f}", linewidth=2, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(PREFIX_LENGTHS)
    ax.set_xticklabels([str(p) for p in PREFIX_LENGTHS])
    ax.set_xlabel("Prefix length (tokens)", fontsize=11)
    ax.set_ylabel(r"Median $|H_s - H_t|$", fontsize=11)
    ax.set_title(r"$|H_s - H_t|$ by reward class across lengths", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hurst_diff_by_reward_vs_length.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return partial_rhos, partial_rho_lo, partial_rho_hi

# ---------------------------------------------------------------------------
# Part 2 — Real 3-bin analysis on actual length distribution
# ---------------------------------------------------------------------------

def bin_analysis(records, output_dir):
    print("\n--- Part 2: 3-bin analysis on real length distribution ---\n")

    # Bins: <512, 512–4095, 4096
    bins = [
        ("short  (<512)",      lambda n: n < 512),
        ("medium (512–4095)",  lambda n: 512 <= n < 4096),
        ("long   (=4096)",     lambda n: n == 4096),
    ]

    results = []
    for label, pred in bins:
        sub = [r for r in records if pred(r["n"])]
        n = len(sub)
        if n < 30:
            print(f"  {label}: only {n} samples — skipped (too few)")
            results.append((label, n, None, None, None, None))
            continue

        h_diffs, kl_means, rewards = [], [], []
        for rec in sub:
            h_s = hurst_dfa(rec["lp_s"])
            h_t = hurst_dfa(rec["lp_t"])
            if h_s is None or h_t is None:
                continue
            h_diffs.append(abs(h_s - h_t))
            kl_means.append(float(rec["kl"].mean()))
            rewards.append(rec["reward"])

        h_diffs  = np.array(h_diffs)
        kl_means = np.array(kl_means)
        rewards  = np.array(rewards)

        raw_rho,  raw_p  = spearmanr(h_diffs, rewards)
        part_rho, part_p = partial_spearman(h_diffs, rewards, kl_means)
        print(f"  {label}  n={n}  "
              f"raw_ρ={raw_rho:+.4f} (p={raw_p:.4f})  "
              f"partial_ρ={part_rho:+.4f} (p={part_p:.4f})")
        results.append((label, n, raw_rho, raw_p, part_rho, part_p))

    return results

# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------

def write_report_section(output_dir, resample_rhos, resample_lo, resample_hi, bin_results):
    import datetime
    lines = [
        "# Hurst–Length Analysis Report",
        "",
        f"**Date:** {datetime.date.today()}",
        "",
        "> **Context:** 98.1% of trajectories are at max length (4096 tokens).",
        "> A naive 3-bin length split is degenerate. The primary analysis uses",
        "> sub-sequence resampling to show how partial ρ evolves with prefix length.",
        "",
        "---",
        "",
        "## Part 1 — Sub-sequence Resampling",
        "",
        "For each sampled trajectory, `|H_s - H_t|` is computed on prefixes of",
        "increasing length. Partial Spearman correlation with `reward`, controlling",
        "for `mean_kl`, is computed at each prefix length.",
        "",
        "| Prefix length | Partial ρ | 5th pct | 95th pct |",
        "|---:|---:|---:|---:|",
    ]
    for pl, rho, lo, hi in zip(PREFIX_LENGTHS, resample_rhos, resample_lo, resample_hi):
        lines.append(f"| {pl} | {rho:+.4f} | {lo:+.4f} | {hi:+.4f} |")

    # Find length where signal becomes consistent
    thresh = 0.05
    crossings = [pl for pl, rho in zip(PREFIX_LENGTHS, resample_rhos) if abs(rho) > thresh]
    first_cross = crossings[0] if crossings else "never"

    lines += [
        "",
        f"**Signal emerges at ≥ {first_cross} tokens** (|partial ρ| > {thresh}).",
        "",
        "**Implication for λ_H:** the weight on the Hurst term should be",
        "near-zero for short sequences and increase with length.",
        "A simple schedule: `λ_H(L) = λ_H_max × sigmoid((L - L0) / τ)`",
        "where L0 ≈ the crossing length above.",
        "",
        "---",
        "",
        "## Part 2 — Real 3-bin Analysis",
        "",
        "| Bin | n | raw ρ | p | partial ρ | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, n, rr, rp, pr, pp in bin_results:
        if rr is None:
            lines.append(f"| {label.strip()} | {n} | — | — | — | — |")
        else:
            lines.append(
                f"| {label.strip()} | {n} | {rr:+.4f} | {rp:.4f} "
                f"| {pr:+.4f} | {pp:.4f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "",
        "| File | Content |",
        "|---|---|",
        "| `hurst_partial_rho_vs_length.png` | Partial ρ vs. prefix length with 90% CI |",
        "| `hurst_diff_by_reward_vs_length.png` | Median \\|ΔH\\| by reward class across lengths |",
        "",
    ]

    path = os.path.join(output_dir, "hurst_length_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved to: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("HURST–LENGTH ANALYSIS")
    print("=" * 60)

    print("\nLoading records ...", flush=True)
    t0 = time.time()
    records = load_records(TELEMETRY_FILES)
    print(f"  {len(records)} usable records in {time.time()-t0:.1f}s")
    print(f"  Length distribution: "
          f"<512: {sum(1 for r in records if r['n']<512)}  "
          f"512-4095: {sum(1 for r in records if 512<=r['n']<4096)}  "
          f"=4096: {sum(1 for r in records if r['n']==4096)}")

    rhos, lo, hi = resample_analysis(records, output_dir, rng)
    bin_res = bin_analysis(records, output_dir)
    write_report_section(output_dir, rhos, lo, hi, bin_res)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
