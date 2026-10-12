import asyncio
import numpy as np
import tinker
from tinker import types
from scipy.stats import linregress
from transformers import AutoTokenizer

def calculate_hurst(logprobs):
    if len(logprobs) < 32: return 0.5
    data = np.array(logprobs)
    N = len(data)
    y = np.cumsum(data - np.mean(data))
    scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), num=10).astype(int))
    fluctuations = []
    for s in scales:
        num_windows = N // s
        if num_windows == 0: continue
        rms = np.sqrt(np.mean([np.var(y[i*s:(i+1)*s]) for i in range(num_windows)]))
        fluctuations.append(rms)
    if len(fluctuations) < 2: return 0.5
    H, _, _, _, _ = linregress(np.log(scales), np.log(fluctuations))
    return float(H)

async def run_experiment():
    service_client = tinker.ServiceClient()
    session_id = "6f9acfcf-13a3-5422-864e-a8e72c390cc0"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    # Пытаемся загрузить конкретные номера шагов
    checkpoints = {
        "Step 0": f"tinker://{session_id}:train:0/sampler_weights/000000",
        "Step 6": f"tinker://{session_id}:train:0/sampler_weights/000006"
    }
    
    base_model = "Qwen/Qwen3-8B-Base"
    teacher_model = "Qwen/Qwen3-30B-A3B"
    teacher_client = await service_client.create_sampling_client_async(base_model=teacher_model)

    problem = "Every morning Aya goes for a 9-kilometer-long walk. Speed s km/h takes 4h (inc t min shop). Speed s+2 km/h takes 2.4h (inc t). Find time for s+0.5 km/h inc t."
    prompt_text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    results = {}
    for name, ckpt in checkpoints.items():
        print(f"--- Running {name} ---")
        try:
            client = await service_client.create_sampling_client_async(base_model=base_model, model_path=ckpt)
            tokens = tokenizer.encode(prompt_text)
            res = await client.sample_async(types.ModelInput.from_ints(tokens), types.SamplingParams(max_tokens=512, temperature=0), num_samples=1)
            resp_tokens = list(res.sequences[0].tokens)
            full_input = types.ModelInput.from_ints(tokens + resp_tokens)
            t_lps = await teacher_client.compute_logprobs_async(full_input)
            resp_lps = t_lps[len(tokens):len(tokens)+400]
            results[name] = {"H": calculate_hurst(resp_lps), "mean_logp": np.mean(resp_lps)}
        except Exception as e:
            print(f"Failed to run {name}: {e}")

    if len(results) >= 1:
        print("\n" + "="*45)
        print(f"{'Metric':<15} | {'Value':<12}")
        print("-" * 45)
        for name, data in results.items():
            print(f"{name} Hurst H   | {data['H']:<12.4f}")
            print(f"{name} Mean Logp | {data['mean_logp']:<12.4f}")
            print(f"{name} Conf      | {np.exp(data['mean_logp'])*100:<11.2f}%")
            print("-" * 45)
        print("="*45)

if __name__ == "__main__":
    asyncio.run(run_experiment())
