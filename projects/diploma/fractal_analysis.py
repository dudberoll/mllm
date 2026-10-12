import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress, spearmanr

def dfa(data, scales):
    """Optimized DFA calculation with vectorized detrending."""
    if len(data) < 32: return 0.5
    N = len(data)
    y = np.cumsum(data - np.mean(data))
    fluctuations = []
    for s in scales:
        num_windows = N // s
        if num_windows == 0: continue
        y_cut = y[:num_windows * s].reshape((num_windows, s))
        x = np.arange(s)
        A = np.vstack([x, np.ones(s)]).T
        # Vectorized LSTSQ
        m_c, _, _, _ = np.linalg.lstsq(A, y_cut.T, rcond=None)
        fit = (A @ m_c).T
        rms = np.sqrt(np.mean((y_cut - fit)**2, axis=1))
        fluctuations.append(np.mean(rms))
    
    if len(fluctuations) < 2: return 0.5
    H, _, _, _, _ = linregress(np.log(scales), np.log(fluctuations))
    return float(H)

def run_fractal_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    # Automatic step size detection (standard is 16, 50, or 128)
    # We will buffer 16 records and then check if the step actually has more
    buffer_size = 128
    print(f"Processing large telemetry file: {file_path}")
    
    step_metrics = []
    current_records = []
    step_idx = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                current_records.append(json.loads(line))
            except: continue
            
            # Process in batches of 128 (max expected step size)
            if len(current_records) >= buffer_size:
                h_s_vals, h_t_vals, acc_vals = [], [], []
                for res in current_records:
                    mask = np.array(res['mask'])
                    active_mask = mask > 0
                    if np.sum(active_mask) < 32: continue
                    
                    lp_s = np.array(res['logp_student'])[active_mask]
                    lp_t = np.array(res['logp_teacher'])[active_mask] if res.get('logp_teacher') else []
                    
                    scales = np.unique(np.logspace(np.log10(4), np.log10(len(lp_s)//4), num=8).astype(int))
                    
                    h_s_vals.append(dfa(lp_s, scales))
                    if len(lp_t) >= 32:
                        h_t_vals.append(dfa(lp_t, scales))
                    acc_vals.append(res.get('individual_reward', 0.0))
                
                if h_s_vals:
                    step_metrics.append({
                        "step": step_idx,
                        "h_student": np.median(h_s_vals),
                        "h_teacher": np.median(h_t_vals) if h_t_vals else 0,
                        "accuracy": np.mean(acc_vals)
                    })
                    print(f"Step {step_idx} analyzed. H_s: {step_metrics[-1]['h_student']:.4f}, Acc: {step_metrics[-1]['accuracy']:.4f}")
                    step_idx += 1
                
                current_records = []

    if not step_metrics:
        print("No valid steps found for analysis.")
        return

    # Final Summary and Plotting
    steps = [m['step'] for m in step_metrics]
    h_s = [m['h_student'] for m in step_metrics]
    h_t = [m['h_teacher'] for m in step_metrics]
    acc = [m['accuracy'] for m in step_metrics]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(steps, h_s, 'g-o', label='H Student')
    plt.plot(steps, h_t, 'r-x', label='H Teacher')
    plt.xlabel('Step batch (128 samples each)')
    plt.ylabel('Hurst Exponent')
    plt.title('Fractal Dynamics (Vectorized Analysis)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, acc, 'b-s', label='Accuracy')
    plt.xlabel('Step batch')
    plt.ylabel('Accuracy')
    plt.title('Performance Evolution')
    plt.legend()
    plt.grid(True)

    out_plot = file_path.replace(".jsonl", "_fast_analysis.png")
    plt.savefig(out_plot)
    print(f"\nAnalysis complete! Plot saved to {out_plot}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fractal_analysis.py <path>")
    else:
        run_fractal_analysis(sys.argv[1])
