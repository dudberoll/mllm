"""
Temporal Hurst analysis: where along the trajectory does the signal live?

Key findings from data exploration:
  - \boxed{ (token IDs [59, 79075, 90]) found in 23% of trajectories
  - Of those, 80% have it in window 7 (last 512-token window)
  - boxed → 66% correct; no-boxed → 9-10% correct
  - "reasoning-core vs finalization" ≈ windows 0-6 vs window 7

Three analyses:
  1. Per-window partial ρ      — where along the trajectory is the signal?
  2. Early (0-6) vs Late (7)   — does the last window dominate?
  3. Boxed / no-boxed split    — is the std_diff signal inside or outside boxed subgroup?

Window size: 512 tokens (W=512, 8 windows per 4096-token trajectory)

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

W          = 512
N_WINDOWS  = 8           # 4096 // 512
BOXED_IDS  = [59, 79075, 90]   # \boxed{ in Qwen3-8B-Base tokenizer
MIN_WIN    = 2

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "offline_validation_results"
)

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


def partial_spearman(x, y, Z):
    """Partial Spearman rho(x, y | Z) via OLS residuals. Z is (n, k) array."""
    def resid(a):
        A = np.column_stack([Z, np.ones(len(Z))])
        coeff, *_ = lstsq(A, a)
        return a - A @ coeff
    rho, p = spearmanr(resid(x), resid(y))
    return float(rho), float(p)


def find_last_boxed_window(masked_tokens: list) -> Optional[int]:
    """Return window index (0-7) of last \boxed{ occurrence, or None."""
    n = len(masked_tokens)
    last_pos = None
    for j in range(n - 2):
        if (masked_tokens[j]   == BOXED_IDS[0] and
            masked_tokens[j+1] == BOXED_IDS[1] and
            masked_tokens[j+2] == BOXED_IDS[2]):
            last_pos = j
    if last_pos is None:
        return None
    return last_pos // W

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

            mask   = np.array(d["mask"], dtype=bool)
            n_act  = int(mask.sum())
            if n_act < W * N_WINDOWS:   # require full 8 windows
                continue

            tokens = d["tokens"]
            masked_tokens = [t for t, m in zip(tokens, mask) if m]

            lp_s = np.array(d["logp_student"])[mask]
            lp_t = np.array(d["logp_teacher"])[mask]
            kl   = np.array(d["step_kl"])[mask]

            # per-window diffs
            diffs = []
            valid = True
            for wi in range(N_WINDOWS):
                h_s = hurst_window(lp_s[wi*W:(wi+1)*W])
                h_t = hurst_window(lp_t[wi*W:(wi+1)*W])
                if h_s is None or h_t is None:
                    diffs.append(None)
                else:
                    diffs.append(h_s - h_t)

            n_valid = sum(1 for d_ in diffs if d_ is not None)
            if n_valid < MIN_WIN:
                continue

            boxed_win = find_last_boxed_window(masked_tokens)
            has_boxed = boxed_win is not None

            records.append({
                "diffs"     : diffs,          # list of 8 floats or None
                "mean_kl"   : float(kl.mean()),
                "reward"    : float(d["individual_reward"]),
                "has_boxed" : has_boxed,
                "boxed_win" : boxed_win,       # None or 0-7
                "run"       : run,
            })

    print(f"\n  Loaded {len(records)} records in {time.time()-t0:.1f}s")
    return records

# ---------------------------------------------------------------------------
# Analysis 1: per-window partial ρ
# ---------------------------------------------------------------------------

def analysis_per_window(records, output_dir):
    print("\n[1] Per-window partial ρ\n")

    reward  = np.array([r["reward"]   for r in records])
    kl      = np.array([r["mean_kl"]  for r in records])

    rhos_raw, rhos_part, ps_part, ns = [], [], [], []

    for wi in range(N_WINDOWS):
        vals = np.array([
            r["diffs"][wi] if r["diffs"][wi] is not None else np.nan
            for r in records
        ])
        ok = np.isfinite(vals)
        x, y, z = vals[ok], reward[ok], kl[ok].reshape(-1, 1)

        raw_rho, _ = spearmanr(x, y)
        part_rho, part_p = partial_spearman(x, y, z)
        rhos_raw.append(float(raw_rho))
        rhos_part.append(float(part_rho))
        ps_part.append(float(part_p))
        ns.append(int(ok.sum()))

        sig = "*" if part_p < 0.05 else " "
        print(f"  window {wi}  n={ok.sum():,}  "
              f"raw_ρ={raw_rho:+.4f}  partial_ρ={part_rho:+.4f}{sig}  p={part_p:.4f}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(N_WINDOWS)
    ax.bar(x_pos - 0.2, rhos_raw,  width=0.35, label="raw ρ",     color="#90CAF9", alpha=0.85)
    ax.bar(x_pos + 0.2, rhos_part, width=0.35, label="partial ρ | KL",
           color=["#1565C0" if p < 0.05 else "#EF9A9A" for p in ps_part], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)

    # label significant bars
    for i, (rho, p) in enumerate(zip(rhos_part, ps_part)):
        if p < 0.05:
            ax.text(i + 0.2, rho + 0.003 * np.sign(rho),
                    f"{rho:+.3f}", ha="center",
                    va="bottom" if rho >= 0 else "top", fontsize=7.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"w{i}\ntokens\n{i*W}–{(i+1)*W}" for i in range(N_WINDOWS)],
                       fontsize=7.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_title(r"Per-window $H_s^i - H_t^i$ correlation with reward"
                 "\n(dark blue = partial p < 0.05)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temporal_per_window.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rhos_part, ps_part

# ---------------------------------------------------------------------------
# Analysis 2: early (w0-6) vs late (w7)
# ---------------------------------------------------------------------------

def analysis_early_late(records, output_dir):
    print("\n[2] Early (windows 0-6) vs Late (window 7)\n")

    reward = np.array([r["reward"]  for r in records])
    kl     = np.array([r["mean_kl"] for r in records])

    results = {}
    for label, window_range in [("early_std  (w0–6)", range(0, 7)),
                                 ("late_diff  (w7)  ", [7])]:
        vals = []
        for rec in records:
            diffs_sel = [rec["diffs"][wi] for wi in window_range
                         if rec["diffs"][wi] is not None]
            if len(diffs_sel) < 1:
                vals.append(np.nan)
            elif len(diffs_sel) == 1:
                vals.append(abs(diffs_sel[0]))          # |diff| for single window
            else:
                vals.append(float(np.std(diffs_sel)))   # std for multi-window

        vals = np.array(vals)
        ok   = np.isfinite(vals)
        x, y, z = vals[ok], reward[ok], kl[ok].reshape(-1, 1)

        raw_rho,  raw_p  = spearmanr(x, y)
        part_rho, part_p = partial_spearman(x, y, z)
        results[label] = (part_rho, part_p, int(ok.sum()))

        sig = "*" if part_p < 0.05 else " "
        print(f"  {label}  n={ok.sum():,}  "
              f"raw_ρ={raw_rho:+.4f}  partial_ρ={part_rho:+.4f}{sig}  p={part_p:.4f}")

    # also: full std (w0-7) for comparison
    vals_full = []
    for rec in records:
        d_all = [v for v in rec["diffs"] if v is not None]
        vals_full.append(float(np.std(d_all)) if len(d_all) >= 2 else np.nan)
    vals_full = np.array(vals_full)
    ok = np.isfinite(vals_full)
    x, y, z = vals_full[ok], reward[ok], kl[ok].reshape(-1, 1)
    raw_rho, _ = spearmanr(x, y)
    part_rho, part_p = partial_spearman(x, y, z)
    results["full_std   (w0–7)"] = (part_rho, part_p, int(ok.sum()))
    sig = "*" if part_p < 0.05 else " "
    print(f"  full_std   (w0–7)  n={ok.sum():,}  "
          f"raw_ρ={raw_rho:+.4f}  partial_ρ={part_rho:+.4f}{sig}  p={part_p:.4f}")

    # --- plot ---
    labels_plot = list(results.keys())
    rhos_plot   = [v[0] for v in results.values()]
    ps_plot     = [v[1] for v in results.values()]
    ns_plot     = [v[2] for v in results.values()]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ["#1565C0" if p < 0.05 else "#EF9A9A" for p in ps_plot]
    bars = ax.bar(range(len(labels_plot)), rhos_plot, color=bar_colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels([l.strip() for l in labels_plot], fontsize=9)
    ax.set_ylabel("Partial Spearman ρ | KL")
    ax.set_title("Early (reasoning) vs Late (finalization) windows\n"
                 "(blue = p < 0.05)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rho, p, n in zip(bars, rhos_plot, ps_plot, ns_plot):
        ax.text(bar.get_x() + bar.get_width()/2,
                rho + 0.003 * np.sign(rho),
                f"{rho:+.3f}\n(n={n:,})",
                ha="center", va="bottom" if rho >= 0 else "top", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temporal_early_vs_late.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results

# ---------------------------------------------------------------------------
# Analysis 3: boxed / no-boxed stratification
# ---------------------------------------------------------------------------

def analysis_boxed_split(records, output_dir):
    print("\n[3] Boxed / no-boxed stratification\n")

    # coverage stats
    n_boxed    = sum(1 for r in records if r["has_boxed"])
    n_noboxed  = sum(1 for r in records if not r["has_boxed"])
    acc_boxed  = np.mean([r["reward"] > 0 for r in records if r["has_boxed"]])
    acc_nobox  = np.mean([r["reward"] > 0 for r in records if not r["has_boxed"]])
    print(f"  boxed  : n={n_boxed:,}  accuracy={acc_boxed:.3f}")
    print(f"  noboxed: n={n_noboxed:,}  accuracy={acc_nobox:.3f}\n")

    results = {}
    for group_name, selector in [
        ("boxed",   lambda r: r["has_boxed"]),
        ("no-boxed",lambda r: not r["has_boxed"]),
        ("all",     lambda r: True),
    ]:
        sub = [r for r in records if selector(r)]
        if len(sub) < 50:
            print(f"  {group_name}: too few ({len(sub)}), skip")
            continue

        # compute std_diff (full, w0-7) for each record in group
        vals, kl_vals, rew_vals = [], [], []
        for rec in sub:
            d_all = [v for v in rec["diffs"] if v is not None]
            if len(d_all) >= 2:
                vals.append(float(np.std(d_all)))
                kl_vals.append(rec["mean_kl"])
                rew_vals.append(rec["reward"])

        vals    = np.array(vals)
        kl_arr  = np.array(kl_vals)
        rew_arr = np.array(rew_vals)

        ok = np.isfinite(vals)
        x, y, z = vals[ok], rew_arr[ok], kl_arr[ok].reshape(-1, 1)

        raw_rho,  raw_p  = spearmanr(x, y)
        part_rho, part_p = partial_spearman(x, y, z)
        results[group_name] = {
            "raw_rho": raw_rho, "raw_p": raw_p,
            "partial_rho": part_rho, "partial_p": part_p,
            "n": int(ok.sum()),
        }
        sig = "*" if part_p < 0.05 else " "
        print(f"  {group_name:10s}  n={ok.sum():,}  "
              f"raw_ρ={raw_rho:+.4f}  partial_ρ={part_rho:+.4f}{sig}  p={part_p:.4f}")

    # --- plot: violin of std_diff by reward class, within boxed / no-boxed ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors = {True: "#1565C0", False: "#E53935"}

    for ax, (has_box, label) in zip(axes, [(True, "boxed"), (False, "no-boxed")]):
        sub = [r for r in records if r["has_boxed"] == has_box]
        reward_sub = np.array([r["reward"] for r in sub])
        vals_sub   = np.array([
            float(np.std([v for v in r["diffs"] if v is not None]))
            if sum(1 for v in r["diffs"] if v is not None) >= 2 else np.nan
            for r in sub
        ])
        ok = np.isfinite(vals_sub)
        unique_r = sorted(set(reward_sub[ok]))
        groups = [vals_sub[ok][reward_sub[ok] == rv] for rv in unique_r]
        if all(len(g) > 0 for g in groups):
            vp = ax.violinplot(groups, positions=range(len(unique_r)),
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colors[has_box])
                body.set_alpha(0.6)
        ax.set_xticks(range(len(unique_r)))
        ax.set_xticklabels([f"r={rv:.1f}\n(n={int((reward_sub[ok]==rv).sum()):,})"
                            for rv in unique_r])
        rr = results.get(label, {})
        ax.set_title(
            f"{label}  (n={ok.sum():,})\n"
            f"partial ρ={rr.get('partial_rho', float('nan')):+.3f}  "
            f"p={rr.get('partial_p', float('nan')):.4f}",
            fontsize=9
        )
        ax.set_ylabel(r"std$(H_s^i - H_t^i)$")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(r"std_diff by reward class: boxed vs no-boxed subgroups", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temporal_boxed_split.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(per_window_rhos, per_window_ps, early_late, boxed_split, output_dir):
    import datetime
    peak_win = int(np.argmax(np.abs(per_window_rhos)))

    lines = [
        "# Temporal Hurst Analysis Report",
        "",
        f"**Date:** {datetime.date.today()}",
        "",
        "W=512 tokens per window, 8 windows per trajectory.",
        "",
        "---",
        "",
        "## Analysis 1 — Per-window partial ρ",
        "",
        "| Window | tokens | partial ρ | p | sig |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for wi in range(N_WINDOWS):
        sig = "✓" if per_window_ps[wi] < 0.05 else ""
        lines.append(
            f"| {wi} | {wi*W}–{(wi+1)*W} | "
            f"{per_window_rhos[wi]:+.4f} | {per_window_ps[wi]:.4f} | {sig} |"
        )
    lines += [
        "",
        f"**Peak signal at window {peak_win}** "
        f"(tokens {peak_win*W}–{(peak_win+1)*W}), "
        f"partial ρ = {per_window_rhos[peak_win]:+.4f}",
        "",
        "---",
        "",
        "## Analysis 2 — Early (w0–6) vs Late (w7)",
        "",
        "| Segment | partial ρ | p | n |",
        "|---|---:|---:|---:|",
    ]
    for label, (rho, p, n) in early_late.items():
        sig = " *" if p < 0.05 else ""
        lines.append(f"| `{label.strip()}` | {rho:+.4f}{sig} | {p:.4f} | {n:,} |")

    lines += [
        "",
        "---",
        "",
        "## Analysis 3 — Boxed / no-boxed stratification",
        "",
        "| Group | n | raw ρ | p | partial ρ | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in boxed_split.items():
        sig = " *" if s["partial_p"] < 0.05 else ""
        lines.append(
            f"| {name} | {s['n']:,} | {s['raw_rho']:+.4f} | {s['raw_p']:.4f} "
            f"| {s['partial_rho']:+.4f}{sig} | {s['partial_p']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "| File | Content |",
        "|---|---|",
        "| `temporal_per_window.png` | Per-window raw and partial ρ bar chart |",
        "| `temporal_early_vs_late.png` | Early vs late segment comparison |",
        "| `temporal_boxed_split.png` | std_diff violin: boxed vs no-boxed subgroups |",
        "",
    ]
    path = os.path.join(output_dir, "temporal_hurst_report.md")
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
    print("TEMPORAL HURST ANALYSIS  (W=512, 8 windows)")
    print("=" * 60)

    print("\nLoading and computing per-window DFA ...\n")
    records = load_records(TELEMETRY_FILES)

    n_boxed = sum(1 for r in records if r["has_boxed"])
    print(f"  boxed: {n_boxed} ({100*n_boxed/len(records):.1f}%)  "
          f"no-boxed: {len(records)-n_boxed} ({100*(1-n_boxed/len(records)):.1f}%)\n")

    pw_rhos, pw_ps = analysis_per_window(records, output_dir)
    el_results     = analysis_early_late(records, output_dir)
    bx_results     = analysis_boxed_split(records, output_dir)
    write_report(pw_rhos, pw_ps, el_results, bx_results, output_dir)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
