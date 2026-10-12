"""
Offline validation of composite regularizer descriptors.

Three checks:
  1. Predictive validity  — Spearman(descriptor, reward) + permutation test
  2. Mutual independence  — pairwise Spearman matrix between descriptors
  3. KL-orthogonality     — partial Spearman(descriptor, reward | mean_kl)

Descriptors:
  var_diff   = |Var(logp_s) - Var(logp_t)|
  acf1_diff  = |ACF1(logp_s) - ACF1(logp_t)|
  hurst_diff = |H(logp_s) - H(logp_t)|   (DFA, same as fractal_analysis.py)

Usage:
  python offline_validation.py [output_dir]
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

# Reuse the DFA implementation from the existing codebase
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

SHORT_THRESHOLD = 512   # active tokens; used for Hurst split analysis
MIN_DFA_LEN     = 32    # minimum tokens for a reliable DFA estimate
N_PERMUTATIONS  = 1000  # for permutation test

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "offline_validation_results")

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def acf1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient."""
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = x @ x
    if denom == 0.0:
        return 0.0
    return float((x[:-1] @ x[1:]) / denom)


def hurst_dfa(x: np.ndarray) -> Optional[float]:
    """DFA Hurst exponent. Returns None if the sequence is too short."""
    n = len(x)
    if n < MIN_DFA_LEN:
        return None
    max_scale = n // 4
    if max_scale < 4:
        return None
    scales = np.unique(
        np.logspace(np.log10(4), np.log10(max_scale), num=8).astype(int)
    )
    return _dfa(x, scales)


def extract_features(record: dict) -> Optional[dict]:
    """
    Compute per-trajectory features from a telemetry record.
    Returns None if the sequence is too short to be useful.
    """
    mask    = np.array(record["mask"], dtype=bool)
    n_active = mask.sum()
    if n_active < 2:
        return None

    lp_s = np.array(record["logp_student"])[mask]
    lp_t = np.array(record["logp_teacher"])[mask]
    kl   = np.array(record["step_kl"])[mask]

    # --- variance ---
    var_diff = float(abs(np.var(lp_s) - np.var(lp_t)))

    # --- ACF1 ---
    acf1_diff = float(abs(acf1(lp_s) - acf1(lp_t)))

    # --- Hurst (DFA) ---
    h_s = hurst_dfa(lp_s)
    h_t = hurst_dfa(lp_t)
    hurst_diff = float(abs(h_s - h_t)) if (h_s is not None and h_t is not None) else None

    return {
        "var_diff"   : var_diff,
        "acf1_diff"  : acf1_diff,
        "hurst_diff" : hurst_diff,
        "mean_kl"    : float(kl.mean()),
        "reward"     : float(record["individual_reward"]),
        "n_active"   : int(n_active),
    }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_features(files: list[str]) -> list[dict]:
    features = []
    total = 0
    t0 = time.time()
    for path in files:
        run_name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            lines = f.readlines()
        n = len(lines)
        print(f"  Loading {run_name}  ({n} records) ...", flush=True)
        for i, line in enumerate(lines):
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"    {i}/{n}  ({elapsed:.0f}s elapsed)", flush=True)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            feat = extract_features(rec)
            if feat is not None:
                feat["run"] = run_name
                features.append(feat)
        total += n
    print(f"\nDone. {len(features)}/{total} records passed quality filter "
          f"in {time.time()-t0:.1f}s\n")
    return features

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def spearman_perm_test(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERMUTATIONS,
                       rng: Optional[np.random.Generator] = None):
    """
    Returns (rho, p_value_two_sided_perm).
    p-value = fraction of permuted |rho| >= |observed rho|.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    obs_rho, _ = spearmanr(x, y)
    count = 0
    for _ in range(n_perm):
        rho_perm, _ = spearmanr(x, rng.permutation(y))
        if abs(rho_perm) >= abs(obs_rho):
            count += 1
    return float(obs_rho), (count + 1) / (n_perm + 1)   # +1 Laplace smoothing


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Partial Spearman correlation: rho(x, y | z).
    Regresses z out of both x and y (OLS), then correlates residuals.
    """
    def residuals(a, b):
        A = np.column_stack([b, np.ones(len(b))])
        coeff, *_ = lstsq(A, a)
        return a - A @ coeff

    res_x = residuals(x, z)
    res_y = residuals(y, z)
    rho, p = spearmanr(res_x, res_y)
    return float(rho), float(p)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

DESCRIPTOR_LABELS = {
    "var_diff"   : r"$|\sigma^2_s - \sigma^2_t|$",
    "acf1_diff"  : r"$|{\rm ACF}_1^s - {\rm ACF}_1^t|$",
    "hurst_diff" : r"$|H_s - H_t|$",
}

COLORS = {"var_diff": "#2196F3", "acf1_diff": "#FF9800", "hurst_diff": "#4CAF50"}


def plot_boxplots(data: dict, reward: np.ndarray, out_path: str):
    """Box plots of each descriptor split by reward class."""
    unique_r = sorted(set(reward))
    n_desc = len(DESCRIPTOR_LABELS)
    fig, axes = plt.subplots(1, n_desc, figsize=(5 * n_desc, 5))
    for ax, (key, label) in zip(axes, DESCRIPTOR_LABELS.items()):
        vals = data[key]
        groups = [vals[reward == r] for r in unique_r]
        bp = ax.boxplot(groups, labels=[str(r) for r in unique_r],
                        patch_artist=True, notch=False)
        for patch, r in zip(bp["boxes"], unique_r):
            patch.set_facecolor(COLORS[key])
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("individual_reward")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Descriptor distribution by reward class", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corr_heatmap(corr_matrix: np.ndarray, labels: list[str], title: str, out_path: str):
    """Annotated Spearman correlation heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr_matrix[i, j]) > 0.6 else "black")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_distributions(perm_results: dict, out_path: str):
    """
    Null distribution histogram + observed rho vertical line per descriptor.
    perm_results[key] = {"obs_rho": float, "perm_rhos": array}
    """
    n_desc = len(perm_results)
    fig, axes = plt.subplots(1, n_desc, figsize=(5 * n_desc, 4))
    for ax, (key, res) in zip(axes, perm_results.items()):
        ax.hist(res["perm_rhos"], bins=40, color="gray", alpha=0.6,
                label="Permuted $\\rho$")
        ax.axvline(res["obs_rho"], color=COLORS[key], linewidth=2,
                   label=f"Observed $\\rho={res['obs_rho']:.3f}$")
        ax.axvline(-res["obs_rho"], color=COLORS[key], linewidth=1.5,
                   linestyle="--", alpha=0.5)
        ax.set_title(DESCRIPTOR_LABELS[key], fontsize=10)
        ax.set_xlabel("Spearman $\\rho$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Permutation test: null distributions vs. observed", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_partial_vs_raw(raw_rhos: dict, partial_rhos: dict, out_path: str):
    """Bar chart comparing raw vs partial Spearman with reward."""
    keys   = list(DESCRIPTOR_LABELS.keys())
    labels = [DESCRIPTOR_LABELS[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, [raw_rhos[k] for k in keys], w,
                   label="Raw $\\rho$(desc, reward)", color=[COLORS[k] for k in keys], alpha=0.8)
    bars2 = ax.bar(x + w/2, [partial_rhos[k] for k in keys], w,
                   label="Partial $\\rho$(desc, reward | KL)", color=[COLORS[k] for k in keys],
                   alpha=0.4, hatch="//")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("Raw vs. partial correlation with reward", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate bar values
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003 * np.sign(h),
                f"{h:.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("OFFLINE VALIDATION: composite regularizer descriptors")
    print("=" * 60)

    # ---- 1. Load data ----
    print("\n[1/4] Loading telemetry...\n")
    features = load_all_features(TELEMETRY_FILES)

    # Unpack arrays
    reward   = np.array([f["reward"]    for f in features])
    mean_kl  = np.array([f["mean_kl"]   for f in features])
    n_active = np.array([f["n_active"]  for f in features])

    var_diff   = np.array([f["var_diff"]   for f in features])
    acf1_diff  = np.array([f["acf1_diff"]  for f in features])
    # Hurst: some entries may be None → build masked array
    hurst_raw  = [f["hurst_diff"] for f in features]
    hurst_mask = np.array([v is not None for v in hurst_raw])
    hurst_diff = np.array([v if v is not None else 0.0 for v in hurst_raw])

    descriptors_full = {
        "var_diff"   : var_diff,
        "acf1_diff"  : acf1_diff,
    }
    descriptors_hurst = {
        "hurst_diff" : hurst_diff[hurst_mask],
    }

    print(f"  Total records: {len(features)}")
    print(f"  Records with valid Hurst: {hurst_mask.sum()}  "
          f"({100*hurst_mask.mean():.1f}%)")
    print(f"  Short sequences (<{SHORT_THRESHOLD} tokens): "
          f"{(n_active < SHORT_THRESHOLD).sum()}")
    print(f"  Reward distribution: "
          + "  ".join(f"{r:.1f}→{(reward==r).sum()}" for r in sorted(set(reward))))

    # ---- 2. Check 1: Predictive validity ----
    print("\n[2/4] Check 1: Predictive validity (Spearman + permutation test)\n")
    rng = np.random.default_rng(42)

    check1_results = {}
    perm_store = {}   # for plots

    def run_check1(key, x, y):
        print(f"  {key}:", end=" ", flush=True)
        obs_rho, p_perm = spearman_perm_test(x, y, N_PERMUTATIONS, rng)
        _, p_scipy = spearmanr(x, y)
        # collect permuted rhos for plotting
        perm_rhos = []
        for _ in range(N_PERMUTATIONS):
            rho_p, _ = spearmanr(x, rng.permutation(y))
            perm_rhos.append(rho_p)
        perm_store[key] = {"obs_rho": obs_rho, "perm_rhos": np.array(perm_rhos)}
        check1_results[key] = {"rho": obs_rho, "p_perm": p_perm, "p_scipy": p_scipy}
        sig = "✓ significant" if p_perm < 0.05 else "✗ not significant"
        print(f"rho={obs_rho:+.4f}  p_perm={p_perm:.4f}  p_scipy={p_scipy:.4f}  {sig}")
        return obs_rho

    for key, arr in descriptors_full.items():
        run_check1(key, arr, reward)
    # Hurst: only on valid subset
    h_reward = reward[hurst_mask]
    run_check1("hurst_diff", hurst_diff[hurst_mask], h_reward)

    # Split by length for Hurst
    print("\n  Hurst by sequence length:")
    for label, sel in [("short", n_active < SHORT_THRESHOLD),
                        ("long",  n_active >= SHORT_THRESHOLD)]:
        sel_h = sel & hurst_mask
        if sel_h.sum() < 30:
            print(f"    {label}: too few samples ({sel_h.sum()}), skip")
            continue
        rho, p = spearmanr(hurst_diff[sel_h], reward[sel_h])
        print(f"    {label} ({sel_h.sum()} samples): rho={rho:+.4f}  p={p:.4f}")

    # ---- 3. Check 2: Mutual independence ----
    print("\n[3/4] Check 2: Pairwise descriptor independence\n")
    desc_keys = list(DESCRIPTOR_LABELS.keys())
    desc_arrs = {
        "var_diff"   : var_diff[hurst_mask],
        "acf1_diff"  : acf1_diff[hurst_mask],
        "hurst_diff" : hurst_diff[hurst_mask],
    }
    n_d = len(desc_keys)
    corr_matrix = np.zeros((n_d, n_d))
    for i, ki in enumerate(desc_keys):
        for j, kj in enumerate(desc_keys):
            rho, _ = spearmanr(desc_arrs[ki], desc_arrs[kj])
            corr_matrix[i, j] = rho

    check2_results = corr_matrix
    print("  Spearman correlation matrix:")
    header = "             " + "  ".join(f"{k:>12}" for k in desc_keys)
    print(f"  {header}")
    for i, ki in enumerate(desc_keys):
        row = "  ".join(f"{corr_matrix[i,j]:>12.4f}" for j in range(n_d))
        print(f"  {ki:12s}  {row}")

    for i, ki in enumerate(desc_keys):
        for j, kj in enumerate(desc_keys):
            if j <= i:
                continue
            rho = corr_matrix[i, j]
            note = "(collinear — consider dropping one)" if abs(rho) > 0.85 else "(independent enough)"
            print(f"  rho({ki}, {kj}) = {rho:+.4f}  {note}")

    # ---- 4. Check 3: Orthogonality to KL ----
    print("\n[4/4] Check 3: Partial Spearman (descriptor, reward | mean_kl)\n")
    check3_results = {}
    raw_rhos = {}

    def run_check3(key, x, y, z):
        raw_rho, raw_p = spearmanr(x, y)
        part_rho, part_p = partial_spearman(x, y, z)
        raw_rhos[key] = raw_rho
        check3_results[key] = {
            "raw_rho": raw_rho, "raw_p": raw_p,
            "partial_rho": part_rho, "partial_p": part_p,
        }
        note = ("✓ orthogonal signal" if abs(part_rho) > 0.05 and part_p < 0.05
                else "~ marginal" if abs(part_rho) > 0.05
                else "✗ absorbed by KL")
        print(f"  {key:12s}  raw_rho={raw_rho:+.4f} (p={raw_p:.4f})  "
              f"partial_rho={part_rho:+.4f} (p={part_p:.4f})  {note}")

    for key, arr in descriptors_full.items():
        run_check3(key, arr, reward, mean_kl)
    run_check3("hurst_diff", hurst_diff[hurst_mask], reward[hurst_mask],
               mean_kl[hurst_mask])

    # ---- 5. Plots ----
    print("\n  Saving plots...")

    # 5a. Box plots
    boxplot_data = {
        "var_diff"   : var_diff,
        "acf1_diff"  : acf1_diff,
        "hurst_diff" : hurst_diff,   # zeros where invalid — fine for boxplot
    }
    plot_boxplots(boxplot_data, reward,
                  os.path.join(output_dir, "boxplots_by_reward.png"))

    # 5b. Correlation heatmap (descriptors)
    short_labels = [r"$|\Delta\sigma^2|$", r"$|\Delta ACF_1|$", r"$|\Delta H|$"]
    plot_corr_heatmap(corr_matrix, short_labels,
                      "Pairwise descriptor correlation (Spearman)",
                      os.path.join(output_dir, "descriptor_corr_heatmap.png"))

    # 5c. Permutation test distributions
    plot_permutation_distributions(
        perm_store, os.path.join(output_dir, "permutation_test.png"))

    # 5d. Raw vs partial correlation bar chart
    partial_rhos_vals = {k: v["partial_rho"] for k, v in check3_results.items()}
    plot_partial_vs_raw(raw_rhos, partial_rhos_vals,
                        os.path.join(output_dir, "raw_vs_partial_rho.png"))

    # ---- 6. Markdown report ----
    print("  Writing markdown report...")
    report = _build_report(features, check1_results, check2_results,
                           check3_results, hurst_mask, n_active, reward)
    report_path = os.path.join(output_dir, "offline_validation_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"All results saved to:  {output_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _decision(rho, p, threshold_rho=0.15, threshold_p=0.05):
    if abs(rho) >= threshold_rho and p < threshold_p:
        return "✅ passes"
    elif abs(rho) >= threshold_rho:
        return "⚠️ rho ok, p marginal"
    else:
        return "❌ fails"


def _build_report(features, check1, check2_matrix, check3, hurst_mask, n_active, reward):
    import datetime
    desc_keys = list(DESCRIPTOR_LABELS.keys())

    lines = [
        "# Offline Validation Report: Composite Regularizer Descriptors",
        "",
        f"**Date:** {datetime.date.today()}  ",
        f"**Records:** {len(features)}  (valid Hurst: {hurst_mask.sum()})  ",
        f"**Reward distribution:** "
        + ", ".join(f"{r:.1f} → {(reward==r).sum()}" for r in sorted(set(reward))),
        "",
        "---",
        "",
        "## Check 1 — Predictive Validity",
        "",
        "Spearman correlation of each descriptor with `individual_reward`,",
        "significance via permutation test (1 000 shuffles).",
        "",
        "| Descriptor | ρ | p (perm) | p (scipy) | Decision (\\|ρ\\|>0.15, p<0.05) |",
        "|---|---:|---:|---:|:---|",
    ]
    for key in desc_keys:
        r = check1[key]
        lines.append(
            f"| `{key}` | {r['rho']:+.4f} | {r['p_perm']:.4f} | "
            f"{r['p_scipy']:.4f} | {_decision(r['rho'], r['p_perm'])} |"
        )

    n_pass = sum(1 for k in desc_keys
                 if abs(check1[k]["rho"]) >= 0.15 and check1[k]["p_perm"] < 0.05)
    lines += [
        "",
        f"**Gate:** ≥2 descriptors must pass → "
        f"{'✅ GATE PASSED' if n_pass >= 2 else '❌ GATE FAILED'} "
        f"({n_pass}/3 pass)",
        "",
        "---",
        "",
        "## Check 2 — Descriptor Independence",
        "",
        "Pairwise Spearman between descriptors.  ",
        "Values > 0.85 indicate collinearity (one could be dropped).",
        "",
        "| | var\\_diff | acf1\\_diff | hurst\\_diff |",
        "|---|---:|---:|---:|",
    ]
    for i, ki in enumerate(desc_keys):
        row = " | ".join(f"{check2_matrix[i,j]:+.4f}" for j in range(len(desc_keys)))
        lines.append(f"| `{ki}` | {row} |")

    lines += [
        "",
        "---",
        "",
        "## Check 3 — Orthogonality to Baseline KL",
        "",
        "Partial Spearman: does the descriptor carry signal beyond `mean_step_kl`?",
        "",
        "| Descriptor | raw ρ | p | partial ρ | p | Decision |",
        "|---|---:|---:|---:|---:|:---|",
    ]
    for key in desc_keys:
        r = check3[key]
        dec = _decision(r["partial_rho"], r["partial_p"], threshold_rho=0.05)
        lines.append(
            f"| `{key}` | {r['raw_rho']:+.4f} | {r['raw_p']:.4f} | "
            f"{r['partial_rho']:+.4f} | {r['partial_p']:.4f} | {dec} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "",
        "| File | Content |",
        "|---|---|",
        "| `boxplots_by_reward.png` | Descriptor distribution split by reward class |",
        "| `descriptor_corr_heatmap.png` | Pairwise Spearman heatmap |",
        "| `permutation_test.png` | Null distributions vs. observed ρ |",
        "| `raw_vs_partial_rho.png` | Raw vs. partial correlation bar chart |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    run_analysis(out)
