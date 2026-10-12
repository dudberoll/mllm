"""
Signal decomposition: which component of std_diff carries the reward signal?

Candidates (all computed at W=512, 8 windows per trajectory):
  std_diff   = std_k(H_s^k - H_t^k)          current champion
  mean_abs   = mean_k |H_s^k - H_t^k|         weak baseline
  corr_st    = Pearson corr_k(H_s^k, H_t^k)   shape alignment (high = aligned)
  std_s      = std_k(H_s^k)                   student variability alone
  std_t      = std_k(H_t^k)                   teacher variability alone

For each candidate:
  - raw Spearman with reward
  - partial Spearman controlling for: KL, n_active, var_diff, acf1_diff

Plus: pairwise Spearman matrix between candidates (collinearity check).

Decision logic:
  corr_st best        → signal is shape alignment, support Option 3
  std_s ≈ std_diff    → signal is student-only, teacher not needed in Hurst term
  std_t also strong   → teacher-side dynamics matter, reference stays meaningful
  std_diff best alone → interaction signal, prefer weighting/gating over direct regularizer

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
from scipy.stats import spearmanr, pearsonr
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

W         = 512
N_WINDOWS = 8
MIN_WIN   = 6    # require at least 6 valid windows for corr to be meaningful

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "offline_validation_results"
)

CANDIDATES = ["std_diff", "mean_abs", "corr_st", "std_s", "std_t"]
CANDIDATE_LABELS = {
    "std_diff"  : r"std$(H_s^k - H_t^k)$",
    "mean_abs"  : r"mean$|H_s^k - H_t^k|$",
    "corr_st"   : r"corr$(H_s^k, H_t^k)$",
    "std_s"     : r"std$(H_s^k)$",
    "std_t"     : r"std$(H_t^k)$",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def acf1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = x @ x
    return float((x[:-1] @ x[1:]) / denom) if denom != 0.0 else 0.0


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray):
    """Partial Spearman rho(x, y | Z), Z is (n, k)."""
    def resid(a):
        A = np.column_stack([Z, np.ones(len(Z))])
        coeff, *_ = lstsq(A, a)
        return a - A @ coeff
    rho, p = spearmanr(resid(x), resid(y))
    return float(rho), float(p)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(files):
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

            mask = np.array(d["mask"], dtype=bool)
            n_act = int(mask.sum())
            if n_act < W * N_WINDOWS:
                continue

            lp_s = np.array(d["logp_student"])[mask]
            lp_t = np.array(d["logp_teacher"])[mask]
            kl   = np.array(d["step_kl"])[mask]

            # per-window Hurst
            hs_wins, ht_wins = [], []
            for wi in range(N_WINDOWS):
                h_s = hurst_window(lp_s[wi*W:(wi+1)*W])
                h_t = hurst_window(lp_t[wi*W:(wi+1)*W])
                hs_wins.append(h_s)
                ht_wins.append(h_t)

            # require MIN_WIN valid pairs
            valid_pairs = [(s, t) for s, t in zip(hs_wins, ht_wins)
                           if s is not None and t is not None]
            if len(valid_pairs) < MIN_WIN:
                continue

            hs_arr = np.array([s for s, _ in valid_pairs])
            ht_arr = np.array([t for _, t in valid_pairs])
            diffs  = hs_arr - ht_arr

            # candidates
            std_diff  = float(np.std(diffs))
            mean_abs  = float(np.mean(np.abs(diffs)))
            std_s     = float(np.std(hs_arr))
            std_t     = float(np.std(ht_arr))

            # Pearson corr between h_s and h_t vectors
            if std_s > 1e-10 and std_t > 1e-10:
                corr_st = float(pearsonr(hs_arr, ht_arr)[0])
            else:
                corr_st = np.nan   # degenerate (flat profile)

            # full-sequence covariates for partial correlation
            var_diff  = float(abs(np.var(lp_s) - np.var(lp_t)))
            acf1_diff = float(abs(acf1(lp_s) - acf1(lp_t)))

            records.append({
                "std_diff"  : std_diff,
                "mean_abs"  : mean_abs,
                "corr_st"   : corr_st,
                "std_s"     : std_s,
                "std_t"     : std_t,
                "mean_kl"   : float(kl.mean()),
                "n_active"  : n_act,
                "var_diff"  : var_diff,
                "acf1_diff" : acf1_diff,
                "reward"    : float(d["individual_reward"]),
            })

    print(f"\n  Loaded {len(records)} records in {time.time()-t0:.1f}s")
    return records

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(records):
    reward  = np.array([r["reward"]    for r in records])
    kl      = np.array([r["mean_kl"]   for r in records])
    n_act   = np.array([r["n_active"]  for r in records], dtype=float)
    var_d   = np.array([r["var_diff"]  for r in records])
    acf_d   = np.array([r["acf1_diff"] for r in records])

    Z_full = np.column_stack([kl, n_act, var_d, acf_d])

    results = {}
    for cand in CANDIDATES:
        x_raw = np.array([r[cand] for r in records])
        ok = np.isfinite(x_raw) & np.isfinite(reward)
        if ok.sum() < 100:
            print(f"  {cand}: too few finite values ({ok.sum()}), skip")
            continue
        x, y    = x_raw[ok], reward[ok]
        Z_sub   = Z_full[ok]

        raw_rho,  raw_p  = spearmanr(x, y)
        part_rho, part_p = partial_spearman(x, y, Z_sub)
        results[cand] = {
            "raw_rho"    : float(raw_rho),
            "raw_p"      : float(raw_p),
            "partial_rho": float(part_rho),
            "partial_p"  : float(part_p),
            "n"          : int(ok.sum()),
        }
    return results


def compute_pairwise(records):
    """Spearman correlation matrix between candidates."""
    arrays = {}
    for cand in CANDIDATES:
        arr = np.array([r[cand] for r in records])
        arrays[cand] = arr

    # common finite mask
    ok = np.ones(len(records), dtype=bool)
    for arr in arrays.values():
        ok &= np.isfinite(arr)

    mat = np.zeros((len(CANDIDATES), len(CANDIDATES)))
    for i, ci in enumerate(CANDIDATES):
        for j, cj in enumerate(CANDIDATES):
            rho, _ = spearmanr(arrays[ci][ok], arrays[cj][ok])
            mat[i, j] = float(rho)

    return mat, int(ok.sum())

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_partial_rho_comparison(results, output_dir):
    cands   = [c for c in CANDIDATES if c in results]
    rhos    = [results[c]["partial_rho"] for c in cands]
    ps      = [results[c]["partial_p"]   for c in cands]
    raw_rhos= [results[c]["raw_rho"]     for c in cands]
    labels  = [CANDIDATE_LABELS[c]       for c in cands]

    x = np.arange(len(cands))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars_raw  = ax.bar(x - w/2, raw_rhos, w, label="raw ρ",
                       color="#90CAF9", alpha=0.85)
    bar_colors = ["#1565C0" if p < 0.05 else "#EF9A9A" for p in ps]
    bars_part = ax.bar(x + w/2, rhos, w, label="partial ρ | KL, n, var, acf1",
                       color=bar_colors, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ with reward")
    ax.set_title("Signal decomposition: raw vs partial ρ for each candidate\n"
                 "(dark blue = partial p < 0.05,  red = not significant)",
                 fontsize=10)

    for bar, rho, p in zip(bars_part, rhos, ps):
        ax.text(bar.get_x() + bar.get_width()/2,
                rho + 0.003 * np.sign(rho),
                f"{rho:+.3f}", ha="center",
                va="bottom" if rho >= 0 else "top", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color="#90CAF9", alpha=0.85, label="raw ρ"),
        Patch(color="#1565C0", alpha=0.85, label="partial ρ (p<0.05)"),
        Patch(color="#EF9A9A", alpha=0.85, label="partial ρ (n.s.)"),
    ], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "decomp_partial_rho.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_heatmap(mat, n, output_dir):
    labels = [CANDIDATE_LABELS[c] for c in CANDIDATES]
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = 1.0
    im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CANDIDATES)))
    ax.set_yticks(range(len(CANDIDATES)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(CANDIDATES)):
        for j in range(len(CANDIDATES)):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.65 else "black")
    ax.set_title(f"Pairwise Spearman between candidates\n(n={n:,})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "decomp_pairwise_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def interpret(results, pairwise_mat):
    """Print a structured decision summary."""
    if not results:
        return
    print("\n" + "="*60)
    print("DECISION SUMMARY")
    print("="*60)

    by_partial = sorted(results.items(), key=lambda x: abs(x[1]["partial_rho"]),
                        reverse=True)
    winner = by_partial[0][0]
    print(f"\n  Best candidate by |partial ρ|: {winner}  "
          f"(ρ={results[winner]['partial_rho']:+.4f})")

    std_diff_rho = results.get("std_diff", {}).get("partial_rho", 0.0)
    std_s_rho    = results.get("std_s",    {}).get("partial_rho", 0.0)
    corr_rho     = results.get("corr_st",  {}).get("partial_rho", 0.0)

    # get pairwise rho between std_diff and std_s
    i_sd = CANDIDATES.index("std_diff")
    i_ss = CANDIDATES.index("std_s")
    rho_sd_ss = pairwise_mat[i_sd, i_ss]

    print(f"\n  std_diff  partial ρ = {std_diff_rho:+.4f}")
    print(f"  std_s     partial ρ = {std_s_rho:+.4f}")
    print(f"  corr_st   partial ρ = {corr_rho:+.4f}")
    print(f"  pairwise Spearman(std_diff, std_s) = {rho_sd_ss:+.4f}")

    print("\n  Interpretation:")
    if abs(corr_rho) > abs(std_diff_rho) * 0.9 and results.get("corr_st", {}).get("partial_p", 1) < 0.05:
        print("  → corr_st is competitive: shape alignment signal is real.")
        print("    Option 3 (1 - corr profile) has empirical support.")
    else:
        print("  → corr_st is weaker: shape alignment alone doesn't explain the signal.")
        print("    Option 3 needs more justification before going online.")

    if rho_sd_ss > 0.85:
        print(f"  → std_diff ≈ std_s (ρ={rho_sd_ss:.2f}): signal is largely student-only.")
        print("    Teacher reference in Hurst term may not be essential.")
    elif rho_sd_ss < 0.5:
        print(f"  → std_diff and std_s are weakly correlated (ρ={rho_sd_ss:.2f}).")
        print("    The interaction (student−teacher diff) carries unique signal.")
        print("    Teacher reference is meaningful.")
    else:
        print(f"  → std_diff and std_s have moderate correlation (ρ={rho_sd_ss:.2f}).")
        print("    Partial decomposition is ambiguous; both contribute.")

    std_t_rho = results.get("std_t", {}).get("partial_rho", 0.0)
    if abs(std_t_rho) > 0.05 and results.get("std_t", {}).get("partial_p", 1) < 0.05:
        print(f"  → std_t is also significant (ρ={std_t_rho:+.4f}):")
        print("    Teacher-side Hurst dynamics vary across trajectories and matter.")
    else:
        print(f"  → std_t is not significant (ρ={std_t_rho:+.4f}).")
        print("    Teacher Hurst profile variability doesn't independently predict reward.")

    print()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results, pairwise_mat, n_pairwise, output_dir):
    import datetime
    lines = [
        "# Signal Decomposition Report",
        "",
        f"**Date:** {datetime.date.today()}",
        "",
        "W=512, 8 windows, MIN_WIN=6 valid pairs required.",
        "Partial control: KL, n_active, var_diff, acf1_diff (all simultaneously).",
        "",
        "---",
        "",
        "## Candidate Comparison",
        "",
        "| Candidate | n | raw ρ | p | partial ρ | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cand in CANDIDATES:
        if cand not in results:
            continue
        s = results[cand]
        sig = " *" if s["partial_p"] < 0.05 else ""
        lines.append(
            f"| `{cand}` | {s['n']:,} | {s['raw_rho']:+.4f} | {s['raw_p']:.4f} "
            f"| {s['partial_rho']:+.4f}{sig} | {s['partial_p']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Pairwise Candidate Correlations",
        f"(n={n_pairwise:,} records with all candidates finite)",
        "",
        "| | " + " | ".join(f"`{c}`" for c in CANDIDATES) + " |",
        "|---|" + "|".join("---:" for _ in CANDIDATES) + "|",
    ]
    for i, ci in enumerate(CANDIDATES):
        row = " | ".join(f"{pairwise_mat[i,j]:+.3f}" for j in range(len(CANDIDATES)))
        lines.append(f"| `{ci}` | {row} |")

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "| File | Content |",
        "|---|---|",
        "| `decomp_partial_rho.png` | Raw vs partial ρ for all candidates |",
        "| `decomp_pairwise_heatmap.png` | Pairwise Spearman between candidates |",
        "",
    ]
    path = os.path.join(output_dir, "signal_decomposition_report.md")
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
    print("SIGNAL DECOMPOSITION  (W=512, 8 windows, MIN_WIN=6)")
    print("=" * 60)

    print("\nLoading and computing per-window DFA ...\n")
    records = load_records(TELEMETRY_FILES)

    print("\nComputing statistics ...\n")
    results = compute_stats(records)

    header = f"  {'candidate':12}  {'n':>6}  {'raw_ρ':>8}  {'raw_p':>7}  {'partial_ρ':>10}  {'p':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cand in CANDIDATES:
        if cand not in results:
            continue
        s = results[cand]
        sig = " *" if s["partial_p"] < 0.05 else "  "
        print(f"  {cand:12}  n={s['n']:5,}  "
              f"raw={s['raw_rho']:+.4f} (p={s['raw_p']:.4f})  "
              f"partial={s['partial_rho']:+.4f}{sig} (p={s['partial_p']:.4f})")

    print("\nComputing pairwise correlations ...")
    pairwise_mat, n_pair = compute_pairwise(records)

    interpret(results, pairwise_mat)

    print("Saving plots ...")
    plot_partial_rho_comparison(results, output_dir)
    plot_pairwise_heatmap(pairwise_mat, n_pair, output_dir)
    write_report(results, pairwise_mat, n_pair, output_dir)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
