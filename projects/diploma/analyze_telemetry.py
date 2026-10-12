import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_telemetry(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    kl_values = []
    logp_student_all = []
    logp_teacher_all = []
    
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total sequences analyzed: {len(records)}")

    for idx, record in enumerate(records):
        step_kl = np.array(record['step_kl'])
        logp_s = np.array(record['logp_student'])
        logp_t = np.array(record['logp_teacher'])
        mask = np.array(record['mask'])

        # Применяем маску (берем только те токены, которые участвовали в лоссе)
        active_indices = np.where(mask > 0)[0]
        if len(active_indices) == 0:
            continue

        kl_values.extend(step_kl[active_indices])
        logp_student_all.extend(logp_s[active_indices])
        logp_teacher_all.extend(logp_t[active_indices])

    kl_values = np.array(kl_values)
    logp_student_all = np.array(logp_student_all)
    logp_teacher_all = np.array(logp_teacher_all)

    print("\n--- Summary Statistics (Masked Tokens) ---")
    print(f"Total active tokens: {len(kl_values)}")
    print(f"Mean KL Divergence: {np.mean(kl_values):.4f}")
    print(f"Max KL Divergence:  {np.max(kl_values):.4f}")
    print(f"Min KL Divergence:  {np.min(kl_values):.4f}")
    
    print(f"\nMean Logp Student: {np.mean(logp_student_all):.4f}")
    print(f"Mean Logp Teacher: {np.mean(logp_teacher_all):.4f}")

    # Анализ распределения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(kl_values, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Step KL')
    plt.xlabel('KL Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(logp_student_all, bins=50, alpha=0.5, label='Student', color='orange')
    plt.hist(logp_teacher_all, bins=50, alpha=0.5, label='Teacher', color='green')
    plt.title('Log-probability Distribution')
    plt.xlabel('Logp')
    plt.legend()

    plt.tight_layout()
    output_plot = os.path.join(os.path.dirname(file_path), "analysis_plot.png")
    plt.savefig(output_plot)
    print(f"\nAnalysis plot saved to: {output_plot}")

    # Топ-5 токенов с самым большим KL (пример первого rollout-а)
    if records:
        print("\n--- Worst Token Examples (Highest KL in first sequence) ---")
        first = records[0]
        skl = np.array(first['step_kl'])
        tokens = first['tokens']
        # Здесь мы не можем декодировать токены без токенизатора, 
        # но можем показать их ID и значения
        top_indices = np.argsort(skl)[-5:][::-1]
        for i in top_indices:
            print(f"Token ID: {tokens[i]:6d} | KL: {skl[i]:.4f} | S_logp: {first['logp_student'][i]:.4f} | T_logp: {first['logp_teacher'][i]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_telemetry.py <path_to_telemetry_rollouts.jsonl>")
    else:
        analyze_telemetry(sys.argv[1])
