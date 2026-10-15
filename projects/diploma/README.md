# On-Policy Distillation for Mathematical Reasoning

Research on training a Qwen3-8B-Base student from a Qwen3-32B teacher on DeepMath. The experiments compare on-policy distillation baselines and trajectory-level regularizers based on the Hurst exponent and `std_t`, then analyze reasoning accuracy and token log-probability dynamics.

Start with [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) for the project overview and [`EXPERIMENT_CONFIGS_SUMMARY.md`](EXPERIMENT_CONFIGS_SUMMARY.md) for experiment settings. Main evaluation and analysis scripts include [`baseline_eval_deepmath.py`](baseline_eval_deepmath.py), [`offline_validation.py`](offline_validation.py), and [`fractal_analysis.py`](fractal_analysis.py).
