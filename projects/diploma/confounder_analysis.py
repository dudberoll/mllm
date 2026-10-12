"""
Confounder analysis for windowed std_diff (best result from windowed_hurst_analysis.py).

Four checks:
  1. Sign sanity     — reward coding + KDE/violin of std_diff by reward class
  2. Scatter shape   — empirical CDF, density, and quantile comparison
  3. Per-run         — partial ρ computed separately for each of the 3 runs
  4. Full confounders— partial ρ(std_diff, reward | mean_kl, n_active, var_diff, acf1_diff)

Best combo from previous analysis: W=512, aggregation=std_diff.

Output → offline_validation_results/
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
from scipy.stats import spearmanr, gaussian_kde
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

W        = 512     # window size — best from windowed_hurst_analysis
MIN_WIN  = 2       # minimum complete windows required
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "offline_validation_results"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acf1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = x @ x
    return float((x[:-1] @ x[1:]) / denom) if denom != 0.0 else 0.0


def hurst_window(x: np.ndarray) -> Optional[float]:
    n = len(x)
    if n < 32:
        return None
    max_scale = n // 4
    if max_scale < 4:
        return None
    scales = np.unique(
        np.logspace(np.log10(4), np.log10(max_scale), num=8).astype(int)
    )
    h = _dfa(x, scales)
    return float(h) if np.isfinite(h) else None


def std_diff_windowed(lp_s: np.ndarray, lp_t: np.ndarray, window: int) -> Optional[float]:
    n_win = len(lp_s) // window
    if n_win < MIN_WIN:
        return None
    diffs = []
    for i in range(n_win):
        h_s = hurst_window(lp_s[i*window:(i+1)*window])
        h_t = hurst_window(lp_t[i*window:(i+1)*window])
        if h_s is not None and h_t is not None:
            diffs.append(h_s - h_t)
    if len(diffs) < MIN_WIN:
        return None
    return float(np.std(diffs))


def partial_spearman_multi(x: np.ndarray, y: np.ndarray, Z: np.ndarray):
    """
    Partial Spearman rho(x, y | Z) where Z is a 2-D covariate matrix (n, k).
    Residualises x and y on Z via OLS, then correlates residuals.
    """
    def residuals(a):
        A = np.column_stack([Z, np.ones(len(Z))])
        coeff, *_ = lstsq(A, a)
        return a - A @ coeff
    rho, p = spearmanr(residuals(x), residuals(y))
    return float(rho), float(p)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(files):
    """
    Returns list of dicts with all features needed for analysis.
    Computes std_diff, var_diff, acf1_diff, mean_kl, n_active per record.
    """
    records = []
    t0 = time.time()
    for path in files:
        run = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            lines = f.readlines()
        n = len(lines)
        print(f"  {run}  ({n} records) ...", flush=True)

        for i, line in enumerate(lines):
            if i % 2000 == 0 and i > 0:
                print(f"    {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            mask  = np.array(d["mask"], dtype=bool)
            n_act = int(mask.sum())
            if n_act < W * MIN_WIN:
                continue

            lp_s = np.array(d["logp_student"])[mask]
            lp_t = np.array(d["logp_teacher"])[mask]
            kl   = np.array(d["step_kl"])[mask]

            sd = std_diff_windowed(lp_s, lp_t, W)
            if sd is None:
                continue

            records.append({
                "std_diff"  : sd,
                "var_diff"  : float(abs(np.var(lp_s) - np.var(lp_t))),
                "acf1_diff" : float(abs(acf1(lp_s) - acf1(lp_t))),
                "mean_kl"   : float(kl.mean()),
                "n_active"  : n_act,
                "reward"    : float(d["individual_reward"]),
                "run"       : run,
            })

    print(f"\n  Loaded {len(records)} records in {time.time()-t0:.1f}s")
    return records

# ---------------------------------------------------------------------------
# Check 1: sign sanity
# ---------------------------------------------------------------------------

def check_sign(records, output_dir):
    print("\n[1] Sign sanity check\n")
    reward  = np.array([r["reward"]   for r in records])
    sd      = np.array([r["std_diff"] for r in records])

    unique_r = sorted(set(reward))
    print("  Reward values and counts:")
    for rv in unique_r:
        sel = reward == rv
        print(f"    {rv:+.1f}  n={sel.sum():,}  "
              f"std_diff: median={np.median(sd[sel]):.4f}  "
              f"mean={np.mean(sd[sel]):.4f}")

    # Direction: is higher reward associated with higher or lower std_diff?
    rho, p = spearmanr(sd, reward)
    print(f"\n  Spearman(std_diff, reward) = {rho:+.4f}  p={p:.2e}")
    direction = "higher std_diff → higher reward" if rho > 0 else "higher std_diff → lower reward"
    print(f"  Direction: {direction}")

    # Plot: violin
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Violin
    ax = axes[0]
    data_by_class = [sd[reward == rv] for rv in unique_r]
    vp = ax.violinplot(data_by_class, positions=range(len(unique_r)),
                       showmedians=True, showextrema=False)
    colors = {-0.1: "#F44336", 0.0: "#9E9E9E", 0.9: "#2196F3", 1.0: "#4CAF50"}
    for body, rv in zip(vp["bodies"], unique_r):
        body.set_facecolor(colors.get(rv, "gray"))
        body.set_alpha(0.7)
    ax.set_xticks(range(len(unique_r)))
    ax.set_xticklabels([f"reward={rv:.1f}\n(n={int((reward==rv).sum()):,})"
                        for rv in unique_r])
    ax.set_ylabel(r"std$(H_s^i - H_t^i)$  [W=512]")
    ax.set_title(r"Distribution of std_diff by reward class", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # KDE overlay
    ax = axes[1]
    x_range = np.linspace(sd.min(), np.percentile(sd, 99), 300)
    for rv in unique_r:
        sel = sd[reward == rv]
        if len(sel) < 10:
            continue
        kde = gaussian_kde(sel, bw_method="scott")
        ax.fill_between(x_range, kde(x_range), alpha=0.3,
                        color=colors.get(rv, "gray"),
                        label=f"reward={rv:.1f}")
        ax.plot(x_range, kde(x_range), color=colors.get(rv, "gray"),
                linewidth=1.5)
    ax.set_xlabel(r"std$(H_s^i - H_t^i)$")
    ax.set_ylabel("Density")
    ax.set_title("KDE of std_diff by reward class", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sign check: Spearman ρ(std_diff, reward) = {rho:+.4f}  (p={p:.2e})\n"
        f"{direction}",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confounder_sign_check.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rho

# ---------------------------------------------------------------------------
# Check 2: scatter shape (empirical CDF + quantile comparison)
# ---------------------------------------------------------------------------

def check_scatter_shape(records, output_dir):
    print("\n[2] Scatter shape\n")
    reward = np.array([r["reward"]   for r in records])
    sd     = np.array([r["std_diff"] for r in records])

    unique_r = sorted(set(reward))
    # Focus on the two main classes
    main_classes = [rv for rv in unique_r if (reward == rv).sum() > 100]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {-0.1: "#F44336", 0.9: "#2196F3", 0.0: "#9E9E9E", 1.0: "#4CAF50"}

    # Empirical CDF
    ax = axes[0]
    for rv in main_classes:
        vals = np.sort(sd[reward == rv])
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        ax.plot(vals, cdf, color=colors.get(rv, "gray"),
                linewidth=2, label=f"reward={rv:.1f}")
    ax.set_xlabel(r"std$(H_s^i - H_t^i)$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("ECDF: std_diff by reward class", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Quantile–quantile across reward classes
    ax = axes[1]
    if len(main_classes) >= 2:
        rv_lo, rv_hi = main_classes[0], main_classes[-1]
        q = np.linspace(1, 99, 99)
        q_lo = np.percentile(sd[reward == rv_lo], q)
        q_hi = np.percentile(sd[reward == rv_hi], q)
        ax.scatter(q_lo, q_hi, s=15, alpha=0.6, color="#7B1FA2")
        lim_lo = min(q_lo.min(), q_hi.min())
        lim_hi = max(q_lo.max(), q_hi.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1,
                label="y = x (identical distributions)")
        ax.set_xlabel(f"Quantile  reward={rv_lo:.1f}")
        ax.set_ylabel(f"Quantile  reward={rv_hi:.1f}")
        ax.set_title("Q–Q plot of std_diff across reward classes", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Summary: median shift and overlap coefficient
        med_lo = np.median(sd[reward == rv_lo])
        med_hi = np.median(sd[reward == rv_hi])
        print(f"  Median std_diff:  reward={rv_lo:.1f} → {med_lo:.4f}")
        print(f"  Median std_diff:  reward={rv_hi:.1f} → {med_hi:.4f}")
        print(f"  Δ median = {med_hi - med_lo:+.4f}  "
              f"({'higher for correct' if med_hi > med_lo else 'lower for correct'})")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confounder_scatter_shape.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Check 3: per-run partial ρ
# ---------------------------------------------------------------------------

def check_per_run(records, output_dir):
    print("\n[3] Per-run partial ρ\n")
    runs = sorted(set(r["run"] for r in records))

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 2.5), 4))
    run_rhos, run_labels = [], []

    results = []
    for run in runs:
        sub = [r for r in records if r["run"] == run]
        if len(sub) < 50:
            print(f"  {run}: too few records ({len(sub)}), skip")
            continue
        sd  = np.array([r["std_diff"] for r in sub])
        rew = np.array([r["reward"]   for r in sub])
        kl  = np.array([r["mean_kl"]  for r in sub])

        finite = np.isfinite(sd) & np.isfinite(rew) & np.isfinite(kl)
        sd, rew, kl = sd[finite], rew[finite], kl[finite]

        raw_rho,  raw_p  = spearmanr(sd, rew)
        part_rho, part_p = partial_spearman_multi(sd, rew, kl.reshape(-1, 1))

        short_run = run[:40] + ("…" if len(run) > 40 else "")
        sig = "*" if part_p < 0.05 else " "
        print(f"  {short_run}")
        print(f"    n={len(sd):,}  raw_ρ={raw_rho:+.4f} (p={raw_p:.4f})  "
              f"partial_ρ={part_rho:+.4f}{sig} (p={part_p:.4f})")

        run_rhos.append(part_rho)
        run_labels.append(run.replace("onpolicy-deepmath-", "").replace(
            "distill-deepmath-", "")[:35])
        results.append((run, len(sd), raw_rho, raw_p, part_rho, part_p))

    x = np.arange(len(run_rhos))
    bars = ax.bar(x, run_rhos, color="#512DA8", alpha=0.75, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Partial Spearman ρ (std_diff, reward | KL)")
    ax.set_title("Per-run partial correlation: consistency check", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rho in zip(bars, run_rhos):
        ax.text(bar.get_x() + bar.get_width()/2,
                rho + 0.003 * np.sign(rho),
                f"{rho:+.3f}", ha="center",
                va="bottom" if rho >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confounder_per_run.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results

# ---------------------------------------------------------------------------
# Check 4: full confounder set
# ---------------------------------------------------------------------------

def check_full_confounders(records, output_dir):
    print("\n[4] Full confounder analysis\n")

    sd     = np.array([r["std_diff"]  for r in records])
    rew    = np.array([r["reward"]    for r in records])
    kl     = np.array([r["mean_kl"]   for r in records])
    n_act  = np.array([r["n_active"]  for r in records], dtype=float)
    var_d  = np.array([r["var_diff"]  for r in records])
    acf_d  = np.array([r["acf1_diff"] for r in records])

    # Keep only finite rows
    finite = (np.isfinite(sd) & np.isfinite(rew) & np.isfinite(kl)
              & np.isfinite(n_act) & np.isfinite(var_d) & np.isfinite(acf_d))
    sd, rew, kl, n_act, var_d, acf_d = (
        sd[finite], rew[finite], kl[finite],
        n_act[finite], var_d[finite], acf_d[finite]
    )

    # Progressive confounder sets
    configs = [
        ("raw (no control)",               np.zeros((len(sd), 0))),
        ("| KL",                            kl.reshape(-1, 1)),
        ("| KL, n_active",                  np.column_stack([kl, n_act])),
        ("| KL, n_active, var_diff",        np.column_stack([kl, n_act, var_d])),
        ("| KL, n_active, var_diff, acf1",  np.column_stack([kl, n_act, var_d, acf_d])),
    ]

    print(f"  {'Control set':<38}  {'partial ρ':>10}  {'p':>8}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*8}")
    rows = []
    for label, Z in configs:
        if Z.shape[1] == 0:
            rho, p = spearmanr(sd, rew)
        else:
            rho, p = partial_spearman_multi(sd, rew, Z)
        sig = " *" if p < 0.05 else "  "
        print(f"  {label:<38}  {rho:>+10.4f}{sig}  {p:>8.4f}")
        rows.append((label, rho, p))

    # Plot: partial ρ as controls are added
    labels_plot = [r[0].replace("| ", "") for r in rows]
    rhos_plot   = [r[1] for r in rows]
    ps_plot     = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors_bar = ["#F44336" if p >= 0.05 else "#2196F3" for p in ps_plot]
    bars = ax.bar(range(len(rhos_plot)), rhos_plot, color=colors_bar, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Partial Spearman ρ")
    ax.set_title("std_diff partial ρ with reward as confounders are added\n"
                 "(blue = p < 0.05,  red = not significant)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rho, p in zip(bars, rhos_plot, ps_plot):
        ax.text(bar.get_x() + bar.get_width()/2,
                rho + 0.002 * np.sign(rho),
                f"{rho:+.3f}", ha="center",
                va="bottom" if rho >= 0 else "top", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", alpha=0.75, label="p < 0.05"),
                        Patch(color="#F44336", alpha=0.75, label="p ≥ 0.05")],
              fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confounder_full.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rows

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(sign_rho, per_run_results, confounder_rows, output_dir):
    import datetime
    direction = "higher std_diff → higher reward" if sign_rho > 0 else "higher std_diff → lower reward"
    lines = [
        "# Confounder Analysis Report",
        "",
        f"**Date:** {datetime.date.today()}  ",
        f"**Feature:** windowed `std_diff`  (W=512, best from windowed_hurst_analysis.py)",
        "",
        "---",
        "",
        "## Check 1 — Sign Sanity",
        "",
        f"Spearman ρ(std_diff, reward) = **{sign_rho:+.4f}**  ",
        f"→ {direction}  ",
        "",
        "> reward coding: −0.1 = incorrect, 0.9 = correct  (bigger is better)",
        "",
        "---",
        "",
        "## Check 3 — Per-run Partial ρ",
        "",
        "| Run | n | raw ρ | p | partial ρ | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run, n, rr, rp, pr, pp in per_run_results:
        short = run[:45]
        sig = " *" if pp < 0.05 else ""
        lines.append(f"| `{short}` | {n:,} | {rr:+.4f} | {rp:.4f} "
                     f"| {pr:+.4f}{sig} | {pp:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Check 4 — Full Confounders",
        "",
        "| Control set | partial ρ | p |",
        "|---|---:|---:|",
    ]
    for label, rho, p in confounder_rows:
        sig = " *" if p < 0.05 else ""
        lines.append(f"| {label} | {rho:+.4f}{sig} | {p:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "| File | Content |",
        "|---|---|",
        "| `confounder_sign_check.png` | Violin + KDE of std_diff by reward class |",
        "| `confounder_scatter_shape.png` | ECDF + Q–Q plot |",
        "| `confounder_per_run.png` | Per-run partial ρ bar chart |",
        "| `confounder_full.png` | Partial ρ as confounders are added progressively |",
        "",
    ]
    path = os.path.join(output_dir, "confounder_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CONFOUNDER ANALYSIS  (windowed std_diff, W=512)")
    print("=" * 60)

    print("\nLoading and computing features ...\n")
    records = load_data(TELEMETRY_FILES)

    sign_rho = check_sign(records, output_dir)
    check_scatter_shape(records, output_dir)
    per_run  = check_per_run(records, output_dir)
    conf     = check_full_confounders(records, output_dir)
    write_report(sign_rho, per_run, conf, output_dir)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
