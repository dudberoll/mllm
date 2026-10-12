"""
Windowed Hurst analysis.

For each trajectory, split the masked logp sequence into non-overlapping
windows and compute H_s_i, H_t_i per window. Aggregate into three scalars:

  mean_abs  = mean(|H_s_i - H_t_i|)   — average divergence across trajectory
  max_abs   = max( |H_s_i - H_t_i|)   — peak divergence
  std_diff  = std( H_s_i - H_t_i )    — variability of student-teacher gap

Repeat for window sizes W ∈ {256, 512, 1024}. For each (W, aggregation):
  - Spearman(feature, reward)
  - Partial Spearman(feature, reward | mean_kl)

Output: table + plots → offline_validation_results/

Usage:
  python windowed_hurst_analysis.py [output_dir]
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

WINDOW_SIZES   = [256, 512, 1024]
AGGREGATIONS   = ["mean_abs", "max_abs", "std_diff"]
AGG_LABELS     = {
    "mean_abs"  : r"mean $|H_s^i - H_t^i|$",
    "max_abs"   : r"max $|H_s^i - H_t^i|$",
    "std_diff"  : r"std $(H_s^i - H_t^i)$",
}
MIN_WINDOWS    = 2     # skip record if fewer complete windows
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "offline_validation_results"
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def hurst_window(x: np.ndarray) -> Optional[float]:
    """DFA Hurst on a single window. Returns None if too short or degenerate."""
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
    if not np.isfinite(h):
        return None
    return h


def windowed_hurst_diff(lp_s: np.ndarray, lp_t: np.ndarray,
                        W: int) -> Optional[np.ndarray]:
    """
    Split lp_s / lp_t into non-overlapping windows of size W.
    Returns array of signed differences (H_s_i - H_t_i) for each window,
    or None if fewer than MIN_WINDOWS complete windows.
    """
    n_windows = len(lp_s) // W
    if n_windows < MIN_WINDOWS:
        return None

    diffs = []
    for i in range(n_windows):
        chunk_s = lp_s[i*W : (i+1)*W]
        chunk_t = lp_t[i*W : (i+1)*W]
        h_s = hurst_window(chunk_s)
        h_t = hurst_window(chunk_t)
        if h_s is None or h_t is None:
            continue
        diffs.append(h_s - h_t)

    if len(diffs) < MIN_WINDOWS:
        return None
    return np.array(diffs)


def aggregate(diffs: np.ndarray) -> dict:
    abs_diffs = np.abs(diffs)
    return {
        "mean_abs" : float(abs_diffs.mean()),
        "max_abs"  : float(abs_diffs.max()),
        "std_diff" : float(diffs.std()),
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
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
    records = []
    for path in files:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mask = np.array(d["mask"], dtype=bool)
                if mask.sum() < WINDOW_SIZES[0] * MIN_WINDOWS:
                    continue
                records.append({
                    "lp_s"   : np.array(d["logp_student"])[mask],
                    "lp_t"   : np.array(d["logp_teacher"])[mask],
                    "kl"     : np.array(d["step_kl"])[mask],
                    "reward" : float(d["individual_reward"]),
                })
    return records

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all(records):
    """
    Returns results[W][agg] = {"values": np.array, "kl": np.array, "reward": np.array}
    """
    results = {W: {agg: {"values": [], "kl": [], "reward": []}
                   for agg in AGGREGATIONS}
               for W in WINDOW_SIZES}

    t0 = time.time()
    n  = len(records)
    for i, rec in enumerate(records):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)

        mean_kl = float(rec["kl"].mean())
        reward  = rec["reward"]

        for W in WINDOW_SIZES:
            diffs = windowed_hurst_diff(rec["lp_s"], rec["lp_t"], W)
            if diffs is None:
                continue
            agg_vals = aggregate(diffs)
            for agg in AGGREGATIONS:
                results[W][agg]["values"].append(agg_vals[agg])
                results[W][agg]["kl"].append(mean_kl)
                results[W][agg]["reward"].append(reward)

    # Convert to arrays
    for W in WINDOW_SIZES:
        for agg in AGGREGATIONS:
            for key in ("values", "kl", "reward"):
                results[W][agg][key] = np.array(results[W][agg][key])

    print(f"  Done in {time.time()-t0:.1f}s")
    return results

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(results):
    """Returns stats[W][agg] = {raw_rho, raw_p, partial_rho, partial_p, n}"""
    stats = {}
    for W in WINDOW_SIZES:
        stats[W] = {}
        for agg in AGGREGATIONS:
            d = results[W][agg]
            x_raw, y_raw, z_raw = d["values"], d["reward"], d["kl"]
            # Drop any NaN / inf that leaked through
            finite = (np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(z_raw))
            x, y, z = x_raw[finite], y_raw[finite], z_raw[finite]
            dropped = int((~finite).sum())
            if dropped:
                print(f"  [W={W} {agg}] dropped {dropped} non-finite rows")
            raw_rho,  raw_p  = spearmanr(x, y)
            part_rho, part_p = partial_spearman(x, y, z)
            stats[W][agg] = {
                "raw_rho"    : float(raw_rho),
                "raw_p"      : float(raw_p),
                "partial_rho": float(part_rho),
                "partial_p"  : float(part_p),
                "n"          : int(finite.sum()),
            }
    return stats

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmaps(stats, output_dir):
    """Two heatmaps side by side: raw ρ and partial ρ."""
    W_labels   = [str(W) for W in WINDOW_SIZES]
    agg_labels = [AGG_LABELS[a] for a in AGGREGATIONS]

    raw_mat     = np.array([[stats[W][a]["raw_rho"]     for W in WINDOW_SIZES]
                             for a in AGGREGATIONS])
    partial_mat = np.array([[stats[W][a]["partial_rho"] for W in WINDOW_SIZES]
                             for a in AGGREGATIONS])
    p_mat       = np.array([[stats[W][a]["partial_p"]   for W in WINDOW_SIZES]
                             for a in AGGREGATIONS])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(
        axes,
        [raw_mat, partial_mat],
        ["Raw Spearman ρ (feature, reward)",
         "Partial Spearman ρ (feature, reward | KL)"]
    ):
        lim = max(0.1, np.abs(mat).max() * 1.1)
        im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(WINDOW_SIZES)))
        ax.set_yticks(range(len(AGGREGATIONS)))
        ax.set_xticklabels([f"W={w}" for w in W_labels], fontsize=9)
        ax.set_yticklabels(agg_labels, fontsize=9)
        ax.set_title(title, fontsize=10)

        for i, agg in enumerate(AGGREGATIONS):
            for j, W in enumerate(WINDOW_SIZES):
                v = mat[i, j]
                sig = "*" if (title.startswith("Partial") and p_mat[i, j] < 0.05) else ""
                ax.text(j, i, f"{v:+.3f}{sig}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > lim * 0.6 else "black")

    fig.suptitle("Windowed Hurst aggregations vs. reward\n(* = partial p < 0.05)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, "windowed_hurst_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_boxplot(results, stats, output_dir):
    """Box plot for the (W, agg) with strongest |partial_rho|."""
    best_W, best_agg, best_rho = None, None, 0.0
    for W in WINDOW_SIZES:
        for agg in AGGREGATIONS:
            r = abs(stats[W][agg]["partial_rho"])
            if r > abs(best_rho):
                best_rho = stats[W][agg]["partial_rho"]
                best_W, best_agg = W, agg

    d = results[best_W][best_agg]
    values, reward = d["values"], d["reward"]
    unique_r = sorted(set(reward))

    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [values[reward == r] for r in unique_r]
    bp = ax.boxplot(groups, tick_labels=[str(r) for r in unique_r],
                    patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4CAF50")
        patch.set_alpha(0.6)

    s = stats[best_W][best_agg]
    ax.set_title(
        f"Best windowed Hurst: W={best_W}, {AGG_LABELS[best_agg]}\n"
        f"partial ρ = {s['partial_rho']:+.4f}  (p={s['partial_p']:.4f})",
        fontsize=10
    )
    ax.set_xlabel("individual_reward")
    ax.set_ylabel(AGG_LABELS[best_agg])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(output_dir, "windowed_hurst_best_boxplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return best_W, best_agg


def plot_window_trajectories(records, W, output_dir, n_examples=6, seed=42):
    """
    Show per-window H_s_i and H_t_i for a few example trajectories,
    split by reward class. Gives intuition for what windowed Hurst captures.
    """
    rng = np.random.default_rng(seed)
    rewards = np.array([r["reward"] for r in records])
    unique_r = sorted(set(rewards))

    examples = []
    for rv in unique_r:
        idx = np.where(rewards == rv)[0]
        chosen = rng.choice(idx, size=min(n_examples // len(unique_r), len(idx)),
                            replace=False)
        examples.extend([(records[i], rv) for i in chosen])

    n_plots = len(examples)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    colors = {-0.1: "#F44336", 0.0: "#9E9E9E", 0.9: "#2196F3", 1.0: "#4CAF50"}

    for ax, (rec, rv) in zip(axes, examples):
        n_windows = len(rec["lp_s"]) // W
        h_s_vals, h_t_vals = [], []
        for i in range(n_windows):
            h_s = hurst_window(rec["lp_s"][i*W:(i+1)*W])
            h_t = hurst_window(rec["lp_t"][i*W:(i+1)*W])
            if h_s is not None and h_t is not None:
                h_s_vals.append(h_s)
                h_t_vals.append(h_t)

        xs = np.arange(len(h_s_vals))
        ax.plot(xs, h_s_vals, "o-", color=colors.get(rv, "gray"),
                label="student", linewidth=1.5, markersize=4)
        ax.plot(xs, h_t_vals, "s--", color="black",
                label="teacher", linewidth=1.5, markersize=4, alpha=0.6)
        ax.fill_between(xs, h_s_vals, h_t_vals, alpha=0.15,
                        color=colors.get(rv, "gray"))
        ax.set_title(f"reward={rv:.1f}", fontsize=9,
                     color=colors.get(rv, "black"))
        ax.set_xlabel("window index")
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Hurst exponent")
    handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.5, label="student"),
        plt.Line2D([0], [0], color="black", linewidth=1.5,
                   linestyle="--", label="teacher"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"Per-window H_s and H_t  (W={W})", fontsize=11)
    fig.tight_layout()
    out = os.path.join(output_dir, f"windowed_hurst_trajectories_W{W}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(stats, best_W, best_agg, output_dir):
    import datetime
    lines = [
        "# Windowed Hurst Analysis Report",
        "",
        f"**Date:** {datetime.date.today()}",
        "",
        "Aggregations: `mean_abs` = mean|ΔH|, `max_abs` = max|ΔH|, `std_diff` = std(ΔH signed)",
        "",
        "## Full Results Table",
        "",
        "| W | Aggregation | n | raw ρ | p | partial ρ | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for W in WINDOW_SIZES:
        for agg in AGGREGATIONS:
            s = stats[W][agg]
            sig = " *" if s["partial_p"] < 0.05 else ""
            lines.append(
                f"| {W} | `{agg}` | {s['n']:,} "
                f"| {s['raw_rho']:+.4f} | {s['raw_p']:.4f} "
                f"| {s['partial_rho']:+.4f}{sig} | {s['partial_p']:.4f} |"
            )

    best_s = stats[best_W][best_agg]
    lines += [
        "",
        f"**Best combination:** W={best_W}, `{best_agg}`  "
        f"→ partial ρ = {best_s['partial_rho']:+.4f} (p={best_s['partial_p']:.4f})",
        "",
        "## Plots",
        "",
        "| File | Content |",
        "|---|---|",
        "| `windowed_hurst_heatmap.png` | Raw vs partial ρ for all W × aggregation |",
        "| `windowed_hurst_best_boxplot.png` | Box plot for best combination |",
    ]
    for W in WINDOW_SIZES:
        lines.append(
            f"| `windowed_hurst_trajectories_W{W}.png` "
            f"| Per-window H_s / H_t examples at W={W} |"
        )
    lines.append("")

    path = os.path.join(output_dir, "windowed_hurst_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("WINDOWED HURST ANALYSIS")
    print("=" * 60)

    print("\nLoading records ...", flush=True)
    records = load_records(TELEMETRY_FILES)
    print(f"  {len(records)} records loaded\n")

    print("Computing windowed DFA (this takes ~7 min) ...\n")
    results = compute_all(records)

    print("\nComputing statistics ...")
    stats = compute_stats(results)

    print("\nResults:\n")
    header = f"{'W':>6}  {'aggregation':12}  {'n':>6}  {'raw_rho':>8}  {'raw_p':>7}  {'partial_rho':>12}  {'partial_p':>10}"
    print(header)
    print("-" * len(header))
    for W in WINDOW_SIZES:
        for agg in AGGREGATIONS:
            s = stats[W][agg]
            sig = " *" if s["partial_p"] < 0.05 else "  "
            print(f"  W={W}  {agg:12}  n={s['n']:5,}  "
                  f"raw={s['raw_rho']:+.4f} (p={s['raw_p']:.4f})  "
                  f"partial={s['partial_rho']:+.4f}{sig} (p={s['partial_p']:.4f})")
        print()

    print("Saving plots ...")
    plot_heatmaps(stats, output_dir)
    best_W, best_agg = plot_best_boxplot(results, stats, output_dir)
    for W in WINDOW_SIZES:
        plot_window_trajectories(records, W, output_dir)
    write_report(stats, best_W, best_agg, output_dir)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
